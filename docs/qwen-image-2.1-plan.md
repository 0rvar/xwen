# Qwen-Image 2.1 on xwen: research and plan

Working doc, opened 2026-09-21. Research only: no code was written for it and none is
planned in it. It is the substance an engineer implements from later; when an arc ships,
its decisions migrate to a new topic file `docs/decisions/qwen-image.md` (and a line in
[decisions.md](decisions.md)), the story to [log.md](log.md) behind a record, and the
architecture reference to a `docs/qwen-image.md` beside [zimage.md](zimage.md), per the
doc map in [AGENTS.md](../AGENTS.md). The scope rule it inherits: image transformers are
in scope held to correctness bars first, with a secondary time-per-image figure in
[perf-state.md](perf-state.md), and no image arc argues for a hot-path change in the
language models (decisions.md "Diffusion image transformers are in scope, held to
correctness bars first").

Scope, widened by the repo owner on 2026-09-21 after the first draft: image EDITING with
several reference images and `<imageN>` tags in the prompt is IN the plan (it was "later"),
the desktop GUI (image-studio) supports all of it, and a section reads vLLM-Omni's port for
performance ideas, Metal being the only target. The order stays T2I first: editing needs
three components T2I does not (the vision tower, the VAE encoder, real three-axis MRoPE),
so it is a phase after the T2I gates, shaped so the T2I phases need no undoing.

Provenance caveat that runs through the whole doc. The research session's egress proxy
refused `huggingface.co`, `hf-mirror.com` and `modelscope.cn`, so no checkpoint file was
read from its source: configs come from byte-for-byte copies vendored on GitHub, some
verified by git-blob SHA-1 or sha256 against third-party manifests of the HF repo, while
the diffusers, transformers, ComfyUI and sd.cpp sources were read directly. Every fact
resting on a mirror or a snippet is marked **[unverified]** and collected again under
"Open questions and unverified facts". The seven session reports cited are scratch, not
committed ("Sources").

## What Qwen-Image 2.1 is

- **Identity.** `Qwen/Qwen-Image-2.1` on HF and ModelScope, GitHub `QwenLM/Qwen-Image-2.1`
  (a separate repo from `QwenLM/Qwen-Image`), released 2026-09-20 (README News line and
  LICENSE header, https://raw.githubusercontent.com/QwenLM/Qwen-Image-2.1/main/README.md):
  "a unified text-to-image generation and image editing model ... 7B parameters in its
  visual generation component (32 Single-Stream DiT layers)". ONE checkpoint does both:
  editing passes reference images through the same encoder (as VL context) and the same
  DiT (as clean latent tokens in front of the target). The README says "Support up to 10
  reference images"; the pipeline enforces no count; ComfyUI's node exposes 16 autogrow
  slots (`image_1`..`image_16`) with a template note of 10. So 10 is a vendor-stated
  trained limit, not a code assertion, and the engine's cap is a config constant.
- **License: Qwen Research License Agreement, non-commercial only.** LICENSE section 1.i
  defines Non-Commercial as "for research or evaluation purposes only"; 2.a grants the
  license "FOR NON-COMMERCIAL PURPOSES ONLY"; 2.b routes commercial use to a separate
  license from `model-business@notice.qwencloud.com`. Qwen-Image 1.0 was Apache-2.0.
  **This is a decision for the repo owner before Phase 0, not for the implementing
  engineer**: whether a non-commercial checkpoint belongs in the registry, and under
  what README note.
- **Lineage.** 2025-08-04 Qwen-Image, a 20B MMDiT (60 dual-stream blocks, dim 3072,
  Apache-2.0), then Edit, Edit-2509, Layered, Edit-2511, 2512; 2026-02-10 Qwen-Image 2.0,
  API-only **[unverified: snippet]**; 2026-09-20 2.1, the first open weights of 2.x.
- **Siblings.** `Qwen/Qwen-Image-2.1-PE-T2I` and `-PE-I2I` are prompt-enhancer VLMs
  (Qwen3.5-VL 9B finetunes), not encoders. `Comfy-Org/Qwen-Image-2.1` is the single-file
  repack; `leejet/Qwen-Image-2.1-GGUF` the sd.cpp author's quants (Q4_K 4.2 GB, Q8_0
  7.69 GB **[unverified]**). No `-Edit` sibling: one model does T2I and editing. No
  official Lightning, Turbo or fp8 variant **[unverified: absence claim]**.
- **Diffusers.** PR #14804 "Add Qwen-Image 2.1", merged 2026-09-18, commit
  `6256aa7666cedd47443adc8f82da9a10e110b09c`
  (https://github.com/huggingface/diffusers/pull/14804): new classes
  `QwenImage21Transformer2DModel`, `QwenImage21Pipeline` (under `pipelines/qwenimage21/`)
  and `AutoencoderKLQwenImage21`; `transformers>=5.17`. Not in a tagged diffusers release
  as of the research **[unverified]**.
- **Ecosystem.** candle has no `qwen_image` module of any version
  (https://github.com/huggingface/candle/tree/main/candle-transformers/src/models), so
  unlike Z-Image there is nothing to vendor: the transformer is written from diffusers.
  stable-diffusion.cpp has day-0 support ("Sources"). ComfyUI v0.37.0 (2026-09-20) has
  native support, a `QwenImage21Cache` KV-cache node and templates at 25 steps, cfg 1
  (commit 6bfaacc); its `TextEncodeQwenImage21` node takes the references and a
  `resolution` budget (default 1024), and sizes the output from `image_1`.

## The architecture in numbers, against Z-Image-Turbo

Sources: the transformer, pipeline and VAE modules on diffusers `main` at 6256aa7, the
mirrored `transformer/`, `vae/` and `scheduler/` configs (three copies agree, equal to
the class defaults; **[unverified]** as files), and for Z-Image [zimage.md](zimage.md).

| | Qwen-Image 2.1 | Z-Image-Turbo |
| --- | --- | --- |
| parameters (transformer) | 7,115,124,736 (7.12 B), computed from the config; 14.23 GB bf16, matching the Comfy repack's 14.2 GB **[unverified]** | 6,154,908,736 |
| blocks | 32 single-stream, all modulated | 30 layers + 2 noise refiner, modulated; + 2 context refiner, not |
| dim / heads | 4096 = 32 x 128, no GQA | 3840 = 30 x 128, no GQA |
| FFN | SwiGLU 4096 -> 12288 -> 4096 (`mlp_ratio` 3); `proj` is the ungated up, `gate_layer` the SiLU'd one | SwiGLU 10240 |
| block norms | `nn.LayerNorm`, no affine, eps 1e-6 | RMSNorm eps 1e-5 |
| QK-norm | RMSNorm over 128, eps 1e-6, learned multiply-ready weight, before rope, on text tokens too | same form |
| biases | none anywhere | none in the projections |
| modulation | ONE shared `SiLU -> Linear(4096, 16384)` on the model, chunked `[scale1, gate1, scale2, gate2]`; `1 + scale`, no shift, `tanh(gate)`; text tokens read a separate t=0 row | per-block adaLN, same four-vector shape and the same `1 + scale` / `tanh` |
| timestep | `t = sigma`, sinusoid 256 (128 freqs `exp(-ln 1e4 * i/128)`, cos first), x1000 inside the model, `Linear 256->4096, SiLU, Linear 4096->4096`, no biases | `t = 1 - sigma`, `FREQUENCY_EMBEDDING_SIZE` 256 |
| rope | interleaved-pair complex rotation in f32, axes (16, 56, 56), theta 1e4, each axis normalised by its own dim; text `(p, p, p)`; image frame = text length, h and w CENTRED, `[-(H - H//2), H//2)` | interleaved-pair, its own axes and theta |
| sequence | TEXT first, target image LAST; output is the suffix `noise_pred[:, -N:]` | image first, prefix narrow |
| output sign | used as-is: `x += (sigma_next - sigma) * v` | negated |
| `txt_in` | zero-centred RMSNorm over 4096 storing `w - 1`, Linear 4096->4096, GELU-tanh, Linear 4096->4096, no bias; the ONLY `(1 + w)` norm in the model | `cap_embedder`: RMSNorm then one Linear |
| final layer | LayerNorm no-affine `* (1 + scale)`, scale from `Linear(SiLU(temb))`, no shift; `proj_out` 4096 -> 64 over the whole joint sequence | adaLN with SiLU |
| latent | 64 channels at 16x, patch 1, `img_in` Linear 64 -> 4096; packing is a raster flatten | 16 channels at 8x, patch 2 |
| tokens at 1024x1024 | 4096 | 4096 |
| tokens at native 2048x2048 | 16384 | n.a. |
| VAE | `AutoencoderKLQwenImage21`: Wan-2.2-style residual VAE specialised to one frame, 2-D convs on disk, `RMS_norm` = L2 over channels per pixel x sqrt(C), RGBA in and out, 64-ch posterior, per-channel `latents_mean`/`latents_std` (64 each) | Flux VAE, GroupNorm, RGB, `shift_factor`/`scaling_factor` |
| text encoder | Qwen3-VL-8B language tower, hidden 4096, LAST layer BEFORE the final norm, raw ChatML string with a fixed system line, first ~14 tokens dropped, no length cap, left padding | Qwen3-4B, `hidden_states[-2]` = index 35, 512-token cap, no drop |
| scheduler | `FlowMatchEulerDiscreteScheduler`, dynamic exponential shift, base 0.5 at 256 tokens to 0.9 at 8192, `shift_terminal` 0.02, 40 steps, sigmas `linspace(1, 1/40, 40)`; at 1024x1024 `mu` 0.693548, first sigma exactly 1.0, last non-zero exactly 0.02 | static shift 3.0, 8 steps |
| guidance | none by default (`true_cfg_scale` 1.0); when on, plain `uncond + w (cond - uncond)`, no `cond_norm` rescale, no guidance embedding | none |
| attention | BLOCK-CAUSAL: text strictly causal, each image block bidirectional within itself and over everything before it; text and condition tokens modulated from a t=0 row so their K/V is cached across steps (`use_kv_cache` True; the cache changes rounding) | bidirectional over the joint sequence |
| dtype | reference bf16 end to end with f32 islands; the block clips at 65504 under fp16, so fp16 is disqualified as for Z-Image | bf16 weights, f32 activations here |

The `mu` at 2048x2048 is 1.312903 (16384 tokens, past `max_image_seq_len` 8192: the line
extrapolates); the in-source fallbacks `4096 / 1.15` are NOT what ships. The 41-entry
1024x1024 grid is in the math report; the reference dump must reproduce it.

## The text encoder is the dense Qwen3 graph, wider

The `text_encoder/config.json` was VERIFIED verbatim (a vendored copy hashes to the git
blob `581ce4655dde5272e577065dd08b1fda7e97d6e8` a third-party manifest records for the HF
file): `Qwen3VLForConditionalGeneration`, `model_type: qwen3_vl`, under `text_config`: 36
layers, hidden 4096, 32 Q / 8 KV heads, head_dim 128, SwiGLU 12288, vocab 151936,
`rms_norm_eps` 1e-6, `rope_theta` 5e6, `rope_scaling {mrope_interleaved: true,
mrope_section: [24, 20, 20], rope_type: default}`, `attention_bias` false,
`tie_word_embeddings` false, `max_position_embeddings` 262144. Against xwen's Qwen3-4B
([qwen3-dense.md](qwen3-dense.md)) the graph is the same in every form that could have
gone the other way: QK-RMSNorm over [128] before rope, no biases, `repeat_interleave` GQA,
HF-form RMSNorm, silu SwiGLU. The deltas: width (4096/12288 against 2560/9728), theta 5e6
(the Instruct-2507 entry already carries it), an untied `lm_head` (622 M, never needed to
encode), the prefix `model.language_model.layers.N.*` with `model.visual.*` beside it,
and the hidden-state rule below.

**MRoPE collapses to plain full NEoX rope for TEXT-ONLY input, and that is inferred, not
read.** `get_rope_index`'s text branch builds `arange(text_len)` on all three axes
(`modeling_qwen3_vl.py` lines 1008-1014); with identical positions the interleaved
recomposition of axis-1/2 frequencies into axis-0 slots is a no-op, leaving `rotate_half`
over all 128 dims at theta 5e6 (sections sum to 64 = head_dim/2, nothing unrotated).
Both reports reach this by reading the definition; the encoder reference dump must
confirm it numerically before anything relies on it. **True for T2I, false for editing**:
a reference image gets distinct T/H/W ids and the text after it advances from a different
origin (the editing section), so the rope is designed as three axes with sections
[24, 20, 20] from the start, the text-only case being the degenerate one exercised first.

**The tokenizer is the Qwen3 family file.** `processor/tokenizer.json` is 11,422,654
bytes, sha256 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`, the
size and sha two manifests record for `Qwen/Qwen3-4B-Base`, `Qwen/Qwen3-8B` and
`Qwen/Qwen3-1.7B`. So it is `VocabFamily::Qwen3` (src/hub.rs:107-147) and no third
vocabulary. This CONTRADICTS the seams report's section 6, which assumed a VL tokenizer
with added vision specials and a 152064 vocab (Qwen2.5-VL's shape) and concluded a third
family was needed; the vision specials (`<|vision_start|>` 151652, `<|image_pad|>` 151655,
`<|vision_end|>` 151653) are among the Qwen3 vocabulary's 26 added tokens already. Still
to check: the non-Base `Qwen/Qwen3-4B` sha, which
`every_qwen3_release_ships_the_same_tokenizer` pins (src/hub.rs:111-121), and the
pre-tokenizer regex inside the 11.4 MB file **[unverified]**; extending that byte-compare
test to the new file is the check.

**What ships under `text_encoder/`.** Four bf16 shards totalling 17,534,339,488 bytes
(two independent manifests agree byte for byte, **[unverified against HF itself]**),
which the parameter count explains only as the FULL VLM: text stack 8,190,735,360 params
with the untied head, plus a ~576 M vision tower (depth 27, 1152 wide, deepstack at
layers 8/16/24); a text-only headless copy would be about 15.1 GB. The shards are NOT
byte-identical to standalone `Qwen/Qwen3-VL-8B-Instruct` (different split and hashes, a
re-save); whether the VALUES are identical is **[unverified]**, as is Instruct vs Thinking.

**The template, verbatim** (`pipeline_qwenimage21.py` lines 202-219), a raw string
handed to the processor, NOT `apply_chat_template` ("the two tokenize differently and
the checkpoint expects this one"):

```
<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
```

An empty prompt becomes `" "`. There is no `max_sequence_length` in the 2.1 pipeline.
`_drop_idx` is derived at runtime as the token count of the chat-templated system
message alone (lines 221-226), which the Qwen3-VL template renders as exactly
`<|im_start|>system\n{content}<|im_end|>\n`: **14 tokens** (`[151644, 8948, 198, 1092,
30782, 408, 323, 23643, 279, 3897, 9934, 13, 151645, 198]`), computed on a 7.0 MB
re-serialised copy, not the shipped file **[unverified on the exact file]**. Everything
from the second `<|im_start|>` on is kept, INCLUDING the trailing
`<|im_end|>\n<|im_start|>assistant\n`; ComfyUI's rule is "drop everything before the
second 151644". The README's neon-sign prompt is 45 tokens, 31 kept.

**The hidden state is `hidden_states[-1]` WITHOUT the final RMSNorm, and it is a silent
trap.** The pipeline's own comment (lines 297-310): transformers 5.0 ties that entry to
`last_hidden_state`, "so it comes back normalized instead — a third of the signal the
transformer reads, which shows up first in rendered text"; it installs a forward hook on
`text_model.norm` returning the module's input to neutralise the norm for the call.
transformers PR #48087 adds `tie_last_hidden_states=False` for 5.18. So the rule is the
residual after ALL 36 layers, no norm: index 36, against Z-Image's 35. **And the repo's
convention applies the norm at that index**:
`qwen3::stack::plan` (src/qwen3/stack.rs:283) returns `(n_run, apply_norm)`, and
[qwen3-dense.md](qwen3-dense.md) "The layer index convention for `encode`" says index
`n_layer` is "after `layers[35]` and then `output_norm`". A naive
`EncoderSpec { layer: 36 }` returns exactly the normed tensor diffusers hooks around. The
spec must express "depth 36, pre-norm" explicitly (a field on `EncoderSpec`,
src/hub.rs:243-260, or a new plan arm), and the encoder fixture must be dumped through
the hook so the gate catches the normed variant, the first wrong-graph bracket.

**What the loader refuses today, and the extension list.** `Qwen3Config::from_hf`
(src/qwen3/config.rs:142-247) hard-refuses `model_type != "qwen3"` (:143-147),
`tie_word_embeddings` false (:158-163) and any non-null `rope_scaling` (:193-199);
`HfQwen3Config` (:70-101) is flat, no `text_config`; `RopeSpec` (:53-66) has no sections;
`Qwen3Set::open` (src/qwen3/safetensors.rs:299-610) validates names against a
`model.layers.N.*` table and resolves the tokenizer from the directory or a sibling
`tokenizer/` (:291-295), Z-Image's layout and not 2.1's `processor/`. The extension:

1. accept `model_type: qwen3_vl` and read the text params from `text_config`;
2. a tensor prefix `model.language_model.` and an ignore-list for `model.visual.*`;
3. allow untied embeddings on an encode-only open, skipping `lm_head.weight` (the
   `has_lm_head` rule at :255-258 tolerates only a byte-equal head today);
4. read `rope_scaling.mrope_*` with `rope_type: default` into a three-axis `RopeSpec`
   (interleaved sections [24, 20, 20]) whose text-only ids reduce to plain full NEoX,
   refusing any other `rope_type`; the vision ids are Phase 5's, the spec is shaped now;
5. the pre-norm tap at depth 36, as above;
6. a raw-template render beside `prompt_ids` (src/zimage/conditioning.rs:33-56), the
   system-turn drop computed by tokenizing the system block with the SAME tokenizer;
7. an encode-only entry shaped like `zimage-turbo-encoder` (src/hub.rs:568-597):
   `Format::SafeTensors` at `text_encoder/config.json`, `tokenizer:
   processor/tokenizer.json`, `allow_zero_runs: &[]`, `servable` false, `auto_fetch`
   false, `trained_context` 262144, `vocab_family` Qwen3.

The seams report's alternative, a `src/qwen3vl/` beside `src/qwen3/`, predates the
tokenizer and template facts; with the graph identical for text it would be a second copy
of the one graded encoder, which the repo refused once already (decisions.md "Text
conditioning comes from `XwenModel::encode`"). The vision tower stays unloaded for T2I;
Phase 5 loads it behind the same entry as a module beside the text stack, not a second
encoder.

## The transformer reuses the Z-Image seams

Every kernel the Z-Image transformer runs on applies unchanged (seams report section 8):

- `ops::matmul_bf16` (src/ops/bf16.rs:7-23): bf16 `[n_out, k]` weight x f32 `[t, k]`,
  `k % 32 == 0` and `n_out % 4 == 0` (dispatch.rs:1792-1797); 4096, 12288 and 64 satisfy
  it. `Projection::project` (src/zimage/linear.rs:171-206) takes rank 2 or 3 and
  flattens, so an `[M, K]` of text-plus-image rows is what it wants.
- `ops::flash_attn_tensor` (src/ops/flash_t.rs:15-27): head_dim exactly 128 (satisfied),
  rank 3 with no batch axis, GQA-capable, bidirectional by construction (next section).
- `ops::rope_pair` (src/ops/rope_pair.rs:6-14): interleaved-pair, any head_dim, tables
  `[seq, head_dim/2]`; the same rotation family with new tables (axes 16/56/56, theta
  1e4, per-axis normalisation, centred and NEGATIVE h/w ids, built rather than indexed
  from a cached 9216-row table).
- `BlockNorm::forward_scaled` (src/zimage/transformer.rs:879-883) folds `1 + scale` into
  the norm weight; it applies because 2.1 has no shift term, BUT it is RMSNorm and 2.1's
  block norms are affine-free LayerNorm, so a LayerNorm variant of the fold is new code,
  at the 0.9 dB-class rounding cost the Z-Image arc accepted (decisions.md "The adaLN
  scale folds into the norm weight").
- `ops::gated_residual` (src/ops/gated_residual.rs:12) and `ops::silu_mul`
  (src/ops/silu_mul.rs:14): bitwise, generic; the block is `h + tanh(gate) * y` twice.
- `ensure_weights_fit_f16` (src/zimage/linear.rs:242-283): the tensor gemm stages weights
  to half, so `|w| > 65504` is refused at load; 2.1's weight range and its sub-normal
  band (linear.rs:10-14) are unknown until the shards are open.

What is new, all model code and no kernel code: the single shared modulation (one
`[B+1, 16384]` matmul per step, cheaper than Z-Image's per-block adaLN, the t=0 row
selected per token); the text-first layout and suffix narrow (transformer.rs:1604 and
:1644 are image-first and prefix); LayerNorm blocks; `txt_in` with its zero-centred norm
(a `NormVariant` beyond `Standard`; src/qwen3/config.rs:40-43 says a Gemma-form norm is a
new arm, never a default); the as-is output sign (pipeline.rs:759-761 negates); `img_in`
at patch 1 with a raster pack; the `(1 + scale)` final LayerNorm; and `hidden_dim()`
(transformer.rs:324-326), a Z-Image `(dim/3)*8` formula where 2.1's is `mlp_ratio * dim`.

## Block-causal attention decomposes onto the existing kernels

The mask (`transformer_qwenimage21.py` lines 257-306) is
`allowed = (q_idx >= kv_idx) OR same_image_block`, ANDed with the key-padding mask, and
for text-to-image at batch 1 there is no padding mask (`encode_prompt` returns `None`
when the mask is all ones, pipeline lines 384-385). With one image block the sequence is
`[text (T) ; image (N)]`: text queries attend causally to text keys and never to the
image; image queries attend to every text and image key, bidirectionally. diffusers'
non-flex processor does exactly this, one SDPA per prefix segment plus one full call
(lines 462-548).

So step 0 is two dispatches of kernels the repo has:

1. the text rows: a causal call over T tens of tokens through the language models'
   `ops::flash_attn` (same operand contract) or candle's masked sdpa; negligible either
   way and not worth a kernel choice;
2. the image rows: `ops::flash_attn_tensor` with `q = [32, N, 128]` and
   `k, v = [32, T + N, 128]`, the text K/V concatenated with the image K/V.

The kernel takes `seq` and `K` as independent extents. Its docstring says every query
sees every key, and its test table (src/ops/flash_t.rs:143-150) includes a "fewer queries
than keys" case (4 heads, seq 40, K 64) and a GQA case, so this is a plain call of the
shipped kernel with no new mask code and no new kernel. One correction to language
elsewhere in this repo's docs: the "queries placed at absolute position K so both mask
tests go vacuous" trick belongs to the OLDER steel arm `ops::flash_attn_bidirectional`
(src/ops/flash.rs:46), the `flash` bisect arm; the shipped `tensor` arm is genuinely
bidirectional (decisions.md "The shipped attention arm is a Metal-4 tensor-op kernel").
K and V enter it as f16 (q is staged half inside), so the kept text K/V is f16 by
construction, the rounding the Z-Image path already accepts: rel L2 1.6e-4 to 4.5e-4
against f32 sdpa, image PSNR 46.54 dB over eight steps.

The cache follows from two facts together: the text rows depend only on text rows, and
they are modulated from the fixed t=0 row in every block and in `norm_out`
(`_select_modulation_rows`, lines 238-254), so the text stream is bit-identical at every
step. Compute each block's post-QK-norm, post-rope text K and V once at step 0, keep them
(32 blocks x 2 x T x 4096 f16, a few MB), and on every later step run ONLY the N image
tokens through the 32 blocks against `cat([kept, fresh])`. That is precisely diffusers'
`use_kv_cache=True` (`"extract"` at step 0, `"cached"` after, lines 351-363 and 951-980),
on by default, and the mode to replicate because the reference sample is produced under
it. Two verification items: the cache is a rounding-level change per diffusers' own
docstring ("Fix a sample by fixing this flag"), so the step-0 gate is unaffected and the
final-image comparison must be dumped with `use_kv_cache=True` on the reference side; and
a "no cache" bisect arm is NOT a wrong-graph bracket, being the same math. With reference
images the prefix is text PLUS condition tokens (next section).

## Image editing: references, tags and the vision tower

Source: the edit-path report (`qwen-image-edit-path.md`, "Sources"): the 2.1 pipeline,
transformer and their tests, transformers' `modeling_qwen3_vl.py`, `processing_qwen3_vl.py`,
`qwen2_vl/image_processing_qwen2_vl.py` and `vision_utils.py`, ComfyUI's `nodes_qwen.py`
and `text_encoders/{qwen_image21,qwen3vl}.py`.

**The surface.** `image=` on `__call__` is a PIL image or numpy array or a FLAT list, one
set shared by every prompt (nested lists and latent tensors raise). The pipeline inserts
the tags itself: `<image1><|vision_start|><|image_pad|><|vision_end|>` before the user
text, N such blocks space-joined, no space before the prompt; ComfyUI renders the same
string. The user MAY write `<image1>` in the text (ComfyUI's template prompt does); the
README's examples use none ("These three characters are sitting around a campfire in a
forest" over three images); the model reads the images in list order either way, and
`<imageN>` is ordinary text as far as every consumer shows **[unverified]**. Output size:
the LAST image's aspect at area `output_resolution^2` (default 1024), multiples of 32,
explicit `height`/`width` winning; ComfyUI sizes from `image_1`, and the README's table
says 2048x2048 **[unverified]**. 40 steps, no CFG, `use_kv_cache`.

**One resize rule.** Every condition image is converted to RGBA and resized ONCE to about
1 MP at its own aspect, both sides multiples of 32, PIL lanczos. The same pixels feed the
VL tower composited over white ("trained with the alpha composited over white for the
vision encoder") and the VAE with all four channels kept, an RGB source getting alpha 1.0.

**The slot rule is a count, not a geometry.** The VL processor's `smart_resize` factor is
`patch_size * merge_size` = 32, the identity on a 32-multiple input unless `min_pixels` /
`max_pixels` bite (class defaults 3136 and 1,003,520, the latter BELOW 1024^2, so the
shipped `preprocessor_config.json` must raise it or a 1024x1024 reference would shrink and
the assertion below fire **[unverified]**). Patchify: 16 px patches in 2x2 merge-window
order, the still image duplicated into both temporal taps. The one `<|image_pad|>` expands
to `(h/32)(w/32)` tokens, 1024 at 1024x1024; `<|vision_start|>` 151652 and `<|vision_end|>`
151653 stay one token each. The VAE encodes the same image at 16x to `(h/16)(w/16)`
latent tokens, 4x the slots. In the DiT, `img_mask` (True at pad positions after the
system drop, plus `target_tokens/4` ones appended) is `repeat_interleave`d by 4 at True
positions and the packed clean latents (raster flatten through `img_in`, condition images
first, target last) are written INTO those positions: slot k receives latent tokens
4k..4k+3, the spatial meaning coming only from the rope built from `img_shapes`. So the VL
image-token hidden states are OVERWRITTEN and never reach the DiT, only the text
positions' states do, and the tie is asserted: `build_token_metadata` raises "img_shapes
accounts for N image tokens but image_pad_mask marks M".

**Block ids and the mask with k images.** `image_ids` is -1 at text and `0..k` at image
positions in order, boundaries from `img_shapes` and NOT from runs of True (adjacent
condition images stay separate blocks). `allowed = (q >= kv) OR same_image_block`: text
token-causal, every image block bidirectional within itself, a later block sees everything
before it, never the reverse. The condition latents are the same clean tensor at every
step, never noised, and under `causal_condition` they and the text read the t=0
modulation row, so the whole prefix `[text ; condition images]` is step-independent and
cached as ONE prefix (`extract` at step 0, `cached` after): 2k+2 calls of the kernels the
block-causal section names at step 0, one call with K/V = `[prefix ; target]` after.

**The rope walk.** One shared `position` counter: a text token gets `frame = h = w =
position` and advances it by one; an image block `(1, H, W)` gets `frame = position` for
all its tokens, `h` in `[-(H - H//2), H//2)` and `w` likewise, centred so a block's spatial
ids do not depend on where it sits, then `position += max(H, W)`. Each image, condition
or target, is its own frame; no 0/1/2 frame index as in Qwen-Image 1.0.

**MRoPE becomes real in the text tower.** `get_rope_index` splits the sequence into runs
by `mm_token_type_ids`: a text run of length L gets `arange(L) + pos` on all three axes
and `pos += L`; an image run with VL grid `(gh, gw) = (h/16, w/16)` gets T = pos, H =
`pos + arange(gh/2)`, W = `pos + arange(gw/2)` meshgridded, then `pos += max(gh, gw)/2`,
where text resumes; `<|vision_start|>`/`<|vision_end|>` are text tokens. The rotary is
INTERLEAVED, `mrope_section` [24, 20, 20] over the 64 frequency slots of head_dim 128:
slots start as T, 1, 4, ..., 58 take H, 2, 5, ..., 59 take W, 60..63 stay T, then
`cat(freqs, freqs)` and NEoX `rotate_half`, theta 5e6. So the encoder section's collapse
holds for T2I and not for editing. The hidden-state rule is unchanged: pre-final-norm
`hidden_states[-1]` through the hook, one call for both (`test_prompt_embeds_are_pre_norm`).

**The Qwen3-VL-8B vision tower in numbers** (transformers `Qwen3VLVisionConfig` defaults
and ComfyUI's `qwen3vl_8b` dict agree; the checkpoint config is **[unverified]**): hidden
1152, 16 heads x 72, intermediate 4304, depth 27, `deepstack_visual_indexes` [8, 16, 24],
patch 16, temporal patch 2, merge 2, `out_hidden_size` 4096; 576,388,336 parameters
computed. Patch embed `Conv3d(3, 1152, kernel (2, 16, 16), bias=True)`, both temporal
taps the same pixels. Position embed `Embedding(2304, 1152)`, a 48x48 grid, bilinear with
`align_corners=True` to each image's `(gh, gw)` in merge-block order, added before the
blocks. 27 pre-LN blocks, `LayerNorm(1152, eps 1e-6)` with affine, full bidirectional
attention per image, `qkv Linear(1152, 3456, bias)`, MLP 1152 -> 4304 -> 1152 with
biases, `gelu_pytorch_tanh` **[unverified]**, not gated. Rope axial 2D over the whole 72:
18 inverse frequencies `theta^-(arange(0, 36, 2)/36)` (theta 10000 in code
**[unverified]**), `(h, w)` ids in merge-block order, layout `[h18 | w18 | h18 | w18]`,
`rotate_half` in fp32. Merger `LN(1152)` -> `view(-1, 4608)` (four CONSECUTIVE rows = one
2x2 window) -> `fc 4608 -> 4608` -> exact `GELU` -> `fc -> 4096`, one token per pad slot.
DeepStack: three more mergers with a post-shuffle `LN(4608)` on the outputs of ViT blocks
8/16/24, each ADDED at image positions to the text tower after decoder layers 0/1/2.
Porter traps, each silent: merge-block token order everywhere (raster patches run and
return garbage); both temporal taps; the ViT rope layout against the LLM's interleaved
[24, 20, 20]; `align_corners=True`; tanh-GELU in the MLP, exact GELU in the mergers;
DeepStack an add, not a replace; mean/std 0.5 per ComfyUI, not CLIP's **[unverified]**.
Head_dim 72 means `ops::flash_attn_tensor` (head_dim exactly 128) does not apply; candle's
sdpa or the steel arm is the fallback, and the tower is not the cost centre: ~1024 tokens
per 1 MP reference through 411 M block parameters is 0.85 TFLOP per image (computed),
against 14.0 GFLOP x 4096 = 57 TFLOP for one DiT step at 1024x1024.

**What the engine lacks for editing, none of which T2I needs**: the Qwen2-VL-style image
preprocessor; the ViT with pos-embed interpolation and the four mergers; three-axis MRoPE
with vision position ids, the embedding scatter at pad tokens and the DeepStack adds in
the text tower; the 2.1 VAE ENCODER with RGBA in; RGBA-aware image I/O; the joint sequence
with clean condition rows and per-block rope frames; the prefix cache covering text plus
condition tokens. Hence editing is Phase 5, after the T2I gates, on seams Phases 1-3 shape.

## The VAE is a new module

From the math report's C.1-C.3 and the seams report's section 4. The conv class
subclasses `nn.Conv2d`, squeezes the frame axis and RAISES if handed a feature cache, so
the weights are 2-D convs and the video path is gone; the seams report's worry about a
temporal kernel of 3 does not arise, and every `time_conv.*` tensor is dead.

- **Blocks.** `ResidualBlock`: `h = conv_shortcut(x)` (1x1 when dims differ);
  `x = silu(RMS_norm(x)) -> conv3x3 -> silu(RMS_norm) -> conv3x3`; `x + h`. `RMS_norm` is
  Wan's `F.normalize(x.float(), dim=1) * sqrt(C) * gamma`, i.e. `x / max(||x||_2, 1e-12)
  * sqrt(C)` over the CHANNEL axis per pixel in f32, NOT `rsqrt(mean(x^2) + eps)`. Mid
  block: resnet, single-head attention over all HxW with scale `1/sqrt(C)`, resnet.
- **Channel plans.** Encoder `[96, 96, 192, 384, 768, 768]`; decoder
  `[1152, 1152, 1152, 576, 288, 144]` (`decoder_base_dim` 144 x `[8, 8, 8, 4, 2, 1]`),
  `conv_in` 3x3 64 -> 1152, `post_quant_conv` 1x1, `conv_out` 3x3 144 -> 4, `clamp(-1, 1)`.
  Up blocks: 3 resnets, nearest-exact 2x plus conv3x3, plus a `DupUp3D` shortcut.
- **The single-frame degeneracies, the trap of this module.** `AvgDown3D` pads a ZERO
  frame in front at temporal stages, so encoder stages 1-3 have HALF their shortcut
  channels identically zero (`out[2c] = 0`, `out[2c+1] = avgpool(in[c])`); stage 0 is a
  plain 2x2 average. `DupUp3D` with `first_chunk=True` keeps only the LAST temporal
  copy: 1152 -> 1152 is an exact nearest-2x, 1152 -> 576 keeps the ODD input channel of
  each pair (`out[o] = nearest2x(in[2o+1])`), and 576 -> 288 INTERLEAVES a pair's two
  channels into even and odd output rows. A port from the docstrings ("average pool",
  "nearest upsample") is wrong on four of eight shortcut stages and reconstructs a
  plausible image; derive the index arithmetic from the `view/permute/view` as written.
- **Latents.** 64-channel posterior, `mean` the first 64 of the encoder's 128 outputs and
  the only half used; decode applies `z * std + mean` per channel with the 64 constants
  in `vae/config.json` (also the class defaults, `autoencoder_kl_qwenimage21.py` lines
  1003-1134). The initial noise is NOT scaled.
- **RGBA.** Four channels in and out; `postprocess` hands a 4-channel array to
  `Image.fromarray`, so the reference PNG is RGBA (B.5). The first arc writes the RGB
  planes and reports the alpha plane's distance from opaque; the PSNR gate compares RGB
  against RGB. RGBA output with the model card's transparency template is a later item.

What reuses: the fold-the-affine-and-activation-into-the-conv-read idea
(src/ops/group_norm.rs:13-41, src/ops/conv2d_direct.rs:20-35), the `Conv { candle,
direct }` two-arm bisect (`XWEN_ZIMAGE_VAE`, src/zimage/vae.rs:34-79, :183-187), and the
f32-decode decision with its 60 dB bar (decisions.md "The VAE decodes in f32 and bf16 is
refuted"). `conv2d_direct` requires kernel in {1, 3}, `c_in % 8 == 0`, f32, stride 1
(dispatch.rs:7752-7754): every decoder conv passes (c_in in {64, 1152, 576, 288, 144});
the encoder's `conv_in` 4 -> 96 does not. T2I never runs the encoder; editing runs it once
per reference (posterior MEAN, then `(z - mean) / std`), so Phase 3 builds it on the candle
arm beside the decoder. The fold does NOT
transfer as-is: `group_norm_fold` returns `(scale, shift)` per `[B, C]`, while `RMS_norm`
is a per-PIXEL factor times a per-channel `gamma`, so the statistics pass and the fold
shape are new. The mid-block attention runs over 4096 positions at 1024x1024 and 16384
at 2048x2048, the count Z-Image's already runs at 1024x1024.

Plan: land the decoder on candle's conv path first, as the bisect arm and the
correctness gate; then a direct-conv arm priced by a microbench in the
`tests/zimage_microbench.rs` shape. **Decoder cost is unpriced.** Derived only: each of
the four widest stages is about 9.4 TFLOP at 2048x2048 (six 3x3 convs; 1152 channels at
256x256, 1152 at 512x512, 576 at 1024x1024 and 288 at 2048x2048 land at the same figure
because channels halve as pixels quadruple), about 40 TFLOP per decode against roughly 10
at 1024x1024, at about 10 TFLOP/s on the direct kernel against 1.2-4.4 on candle's im2col
chain at Z-Image's shapes (AGENTS.md): tens of seconds on the candle arm at 2048x2048, so
the direct arm is not optional at the native size.

## Pipeline, registry, memory, serve

**Registry** (seams report sections 1, 2, 5). A new `Model` pair, two `Checkpoint` consts,
`MODELS: [Model; 11]` (src/hub.rs:151-161), and an arm in every exhaustive match on
`Model`: `checkpoint` (:639), `safetensors_rope_theta` (:724), `trained_context` (:774),
`vocab_family` (:793), `full_name` (:833), `chat_dialect` (:857),
`recommended_presence_penalty` (:894, "exhaustive on purpose, with no `_` arm", :892-893),
`auto_fetch` (:946), `not_servable_reason` (:991), `supports_drafting` (:1072),
`draft_default_on` (:1110), `Display` and `FromStr`, plus src/sampler.rs:129 and the test
at src/serve/mod.rs:2084-2089. The pipeline entry is `Format::Diffusion { text_encoder }`
(src/hub.rs:235-240), whose doc comment (:222-234) states the rule: the layout lives in
the loader, the entry owns only which files a fetch pulls, `model_index.json` first
(:599-603) so the resolved path's parent is the snapshot root, which
`diffusion_snapshot_root` (src/checkpoint.rs:265-283) recognises. **Doc drift found**:
docs/decisions/zimage.md:36-38 describes `Format::Diffusion` with four fields
(`text_encoder, transformer_config, vae_config, scheduler_config`); the code has one.
Follow the code and fix the paragraph when the new topic file is written. Files
**[unverified: HF tree not listed]**: `transformer/` in two shards
(`diffusion_pytorch_model-0000{1,2}-of-00002.safetensors`), `vae/` (~0.25 GB), the four
`text_encoder/` shards with index and configs, `processor/tokenizer.json` and its config,
`scheduler/scheduler_config.json`, `model_index.json`; about 33.1 GB per a snippet. The
on-disk dtype of `transformer/` and `vae/` (bf16 by every indirect sign) is read off the
safetensors headers at fetch.

**Module.** A new top-level `src/qwen_image/`, NOT under `src/zimage/`: that directory is
a vendored-and-corrected candle module for one model with a never-resync rule
(src/zimage/mod.rs:1-12), and 2.1 is written from diffusers. The `CheckpointSource`
diffusion arm's reopen condition, "when a second consumer needs the transformer or the
VAE opened, build the arm then, with two call sites to shape it"
(docs/decisions/zimage.md:57-60), is now met; `ZImagePipeline::load_cancellable`
(src/zimage/pipeline.rs:203-290), with its hard-coded `transformer/`, `vae/`, `scheduler/`
paths (:217-218, :264-268, :274-275) and single `VarBuilder` (:231-232), is the first.

**Scheduler.** `FlowMatchEulerDiscreteScheduler::new` refuses `use_dynamic_shifting`
outright (src/zimage/scheduler.rs:97-103). 2.1 needs the exponential time shift
`exp(mu) / (exp(mu) + (1/t - 1))` with `mu` from the TARGET token count
(`0.5 + (0.9 - 0.5) / (8192 - 256) * (N - 256)`), then the `shift_terminal` stretch
(`1 - (1 - t) / ((1 - t_last) / (1 - 0.02))`), over `linspace(1, 1/N, N)`, then a
trailing 0. Keep Z-Image's static arm untouched, add the dynamic one as a second arm the
config selects, and validate at load before anything large is resident
(pipeline.rs:274-278).

**Sizes.** `check_size` (pipeline.rs:305-335), `latent_size` (:338-343) and
`inputs::snap_size` (src/zimage/inputs.rs:7-30) encode Z-Image's three rules. 2.1's: the
reference floors each side to a multiple of 32 px BEFORE computing the latent
(`prepare_latents`, pipeline lines 631-633: `height = 2 * (height // 32)`), so the rule
for xwen is multiples of 32 on both sides, refused rather than floored; there is no
token-count multiple (no pad tokens in 2.1); and the rope table's real bound is a side of
2048 latent tokens = 32768 px (positions -1024..8191 per axis, the centred split needing
`H - H//2 <= 1024`), so memory binds long before rope does, though the bound is
re-checked in the forward as Z-Image's is (transformer.rs:1547-1570). Native size is
2048x2048 with the README's aspect table (2400x1792, 2528x1696, 2752x1536 and
transposes); whether 1024x1024 is served by default is a cost decision, not correctness.

**Memory.** `memory::image_peak` (src/memory.rs:654-669) returns a flat 40 GiB and
refuses over 1,048,576 pixels. Neither survives: the peak becomes per model. Resident:
transformer 14.23 GB bf16, encoder 15.1 GB bf16 headless and vision-free (16.4 with the
untied head; editing adds the 576 M vision tower, 1.15 GB bf16, and 2.15 GB of prefix K/V
per 1 MP reference), VAE ~0.25 GB, about 30 GB before activations. Transients at
2048x2048: one
FFN intermediate is 16384 x 12288 x 4 B = 805 MB in f32, two live at once plus the
product, a few GB, and the flash kernel materialises no score matrix. Under 40 GiB on
paper; measured before it is declared, as Z-Image's was.

**Serve.** The images route compares `model` against one constant
(src/serve/images.rs:867-882) and 400s `negative_prompt` and `guidance_scale` with a
sentence about distillation (:936-950). For 2.1 the `model` rule becomes a lookup over
the image-capable entries; `negative_prompt` plus `true_cfg_scale` become a legal opt-in
(two forwards per step, each with its own text cache, `uncond + w (cond - uncond)` at
the `velocity_batched_cancellable` seam, pipeline.rs:740-762) while `guidance_scale`,
the embedding kind, stays a 400. The `/v1/models` exclusion, the `image-engine` thread's
lazy load and idle unload (:580-770), the 403 rule and the queue generalise unchanged.
The 1024x1024 area cap is re-measured for this model, not lifted.

## The GUI, the CLI and the HTTP surface for editing

Source: the GUI report (`xwen-gui-and-edit-surfaces.md`, "Sources"), file:line as of
2026-09-21.

**What image-studio is.** A standalone Bun/Vite/React 19 plus Tauri v2 desktop app under
`image-studio/`, about 7,250 lines (2,986 frontend, 3,486 Rust, 780 Playwright), built by
`just image-studio`; the record is [records/image-studio.md](records/image-studio.md). It
does not link xwen: its Rust side (`src-tauri/src/state.rs:308-352`) talks to serve over
HTTP, `POST /v1/images/render` for every render and `GET /v1/models` (:293) for the chat
sidebar only, images crossing as base64 data URLs in JSON, never multipart
(`domain.ts:113-129`), snapshotted by SHA-256 under `inputs/` (`storage.rs:70-92`). One
`StudioSettings` object owns the editor state (`domain.ts:18-23`, `App.tsx:44-351`).
Modes Text / Image / Inpaint (`SettingsPanel.tsx:43-45`), ONE source image by
construction, mask painting, LoRA, ControlNet, batches with YAML manifests, a chat
sidebar whose one tool is `queue_txt2img` (`chatTurn.ts:7-34`). No model picker: no
`model` field in `src-tauri/src/models.rs:62-81` and none on the render route, the record
saying checkpoint choice is not a GUI selection because the API does not expose it
([records/image-studio.md](records/image-studio.md#application-boundary)).

**The HTTP and CLI surface today.** `RenderRequest` (`src/serve/image_control.rs:61-75`,
`deny_unknown_fields`): `prompt, width, height, steps, seed, n, loras[], init_image,
strength, mask, mask_blur, control{...}`; `image_bytes` takes a data URI, a server-local
path or bare base64 (:78-94); `prepare` (:96-253) funnels the scalars through a
synthesised `ImagesRequest` into `images::validate`, so the Z-Image-only `model` rule and
the `negative_prompt`/`guidance_scale` 400s apply to it too. `ImageInputs`
(`src/serve/images.rs:103-108`) holds exactly one `edit`, the LoRAs and one
`ControlInput`. The multipart `edits`/`variations` routes: `multipart_fields` (:270-295)
aliases `image[]` to `image` and rejects any duplicate field with "this route accepts one
input image" (:288-292). The envelope hard-codes `"model": Model::ZImageTurbo.full_name()`
(`images.rs:1088`). The CLI: `Cmd::Image` (`src/bin/xwen/main.rs:598-646`) with
`ImageControlArgs` (:2396-2424), sizing from the source when `--width/--height` are
omitted (:2489-2492). `inputs::prepare_image` (`src/zimage/inputs.rs:35-46`) drops alpha
(`to_rgb8()`), and `conditioning::prompt_ids` (`src/zimage/conditioning.rs:33-55`) has no
tag or placeholder notion.

**Additions**, engine first:

- (a) serve. `references: [{image, tag?}]` on `RenderRequest` (`image_control.rs:61-75`),
  each decoded and resized in `prepare` (:96-253) by the one resize rule with alpha kept;
  `ImageInputs` (`images.rs:103-108`) widened to a list; `model` selectable on the images
  routes (:867-881, a lookup over the image-capable entries) and echoed in the envelope
  (:1088); `multipart_fields` (`image_control.rs:288-292`) accepting repeated `image[]`
  parts in order when the target model takes references, still one for Z-Image;
  `negative_prompt` and `true_cfg_scale` validated per model (400 on Z-Image, opt-in on
  2.1; `guidance_scale` a 400 on both). A tag defaults to the list position; one that is
  not `imageN` for its position is a 400, the template numbering by order.
- (b) CLI. Repeatable `--reference path[:tag]` (alias `--ref`) in `ImageControlArgs`
  (`main.rs:2396-2424`), the size defaulting from the LAST reference's aspect at 1024^2
  as the pipeline does, `--model qwen-image-2.1`; `--init` stays Z-Image's img2img.
- (c) prompt. The engine inserts the `<imageN><|vision_start|><|image_pad|><|vision_end|>`
  markup itself, in a renderer for this model beside `prompt_ids` (a
  `src/qwen_image/conditioning.rs`), tags numbered by list order and space-joined before
  the user text, which may mention `<imageN>` and passes through verbatim. One render for
  the CLI, the route and the GUI, byte-compared once against the fixture.
- (d) GUI. A model picker fed from a models listing for images: `/v1/models` excludes
  image models and is what the chat sidebar and OpenAI clients read, so the doc's choice
  is a new `GET /v1/images/models` under `is_images_path` (`images.rs:55-67`), listing the
  cached image-capable entries with their defaults (steps, size rule, max references,
  which controls exist); listing them on `/v1/models` would offer chat clients an entry
  the chat routes refuse. A reference-image list (add, remove, reorder, thumbnails,
  drag-and-drop through `useImageDrop.ts`), a tag chip per reference inserting `<imageN>`
  at the prompt caret, mode "Edit" beside Text / Image / Inpaint, size defaulting from
  the last reference, per-model controls (steps default 40 against 8; strength, mask,
  ControlNet and LoRA hidden for 2.1 until they exist for it), the batch planner
  (`domain.ts:214-245`) and YAML manifests carrying N inputs (each distinct data URL
  crossing the bridge once, `batchDraft.ts:20-52`, as today), the Tauri `models.rs:62-81`
  and `state.rs:452-463, 628-699` types and snapshots, and Playwright coverage through
  the fixture bridge (`bridge.ts:103-264`). The chat sidebar's tool stays txt2img-only.

The GUI work is client work with no correctness bar, so it ranks after the engine phases
and is gated by the engine's HTTP contract being frozen (`references`, the models listing,
the multipart rule). It updates [records/image-studio.md](records/image-studio.md) and
[zimage-control-prd.md](zimage-control-prd.md#clients), plus README's GUI and
native-render paragraphs and `image-studio/README.md`.

## Performance inspiration from vLLM-Omni

Source: the vLLM-Omni report (`vllm-omni-qwen-image.md`, "Sources"): an UNMERGED PR,
vllm-project/vllm-omni#7759 "[New model] Qwen-image-2.1 support", head commit
`3ea9e605011fc2a0bfa9f3e8531286fb683068dd` (2026-09-20), read 2026-09-21 with reviewer
blockers open; `main` has no `qwen_image_21` directory. Every number is theirs (GB200,
1024x1024, 50 steps) and none is a Metal figure. Ranked by what transfers to batch 1:

1. **The prefix KV cache**, what the block-causal section already plans, with their
   detail: per block per CFG branch the post-QK-norm, post-rope K and V of the prefix in
   bf16, `.clone()`d at step 0 only so the cache owns its storage instead of pinning the
   prefill tensor. Later steps run ONLY the target tokens: no `txt_in`, no prefix
   attention rows, no prefix FFN; decode attention is one mask-free bidirectional call,
   Q the target rows, K/V `[prefix ; target]`; with 2K condition images the prefix is
   most of the step. A second cache per CFG branch. Their per-block `torch.cat([cached;
   new])` copies the whole `[prefix ; target]` K/V per block per step; a kernel reading
   K/V from two base pointers avoids it, an unpriced change to `flash_t.metal`.
2. **Block-causal prefill as segments, not a mask**: 2k+2 attention calls per block with
   k condition images, no FlexAttention, the dense mask their fallback. Already the plan.
3. **The CUDA-graph analogue.** They capture the whole decode step per exact layout with
   static buffers (key: branch, batch, `prefix_len`, `img_shapes`, dtype, backend; LRU of
   8; prefill eager): 171.2 -> 142.1 ms/step, 8.70 -> 7.24 s (1.20x) at CFG 4; after
   #7790, T2I 8,863 -> 7,958 ms and two-reference editing 14,043 -> 10,380 ms. Launch and
   allocation overhead, not arithmetic. The Metal analogue: one command buffer per step
   with no intermediate `synchronize()`, and per-layout preallocated activations and
   `[prefix ; target]` K/V scratch instead of the buffer pool first-touching each dispatch.
   Unpriced on Metal; the figure to price it is CPU encode time plus the gaps between
   command buffers per step, never their ratio; the repo already knows every profiler
   mark syncs AND evicts candle's buffer pool, running a table 1.39x high
   ([AGENTS.md](../AGENTS.md), the `profile.rs` paragraph).
4. **Fusions: none beyond fused QKV**; norms, `(1+scale)`, `tanh(gate)` residual, SwiGLU
   and rope are stock eager kernels the graph hides. `rope_pair`,
   `BlockNorm::forward_scaled`, `gated_residual` and `silu_mul` already cover them.
5. **Batching the two CFG branches into B=2** when CFG is on: they run cond and uncond
   serially at CFG world size 1. Unpriced, and only when CFG is served.

Does not transfer, one line each: step-level and phase-aware batching (wave batching of
prefill against decode, zero at batch 1 by construction); TP, Ulysses SP, distributed VAE
and CPU offload (multi-GPU, or a fit for GPUs that cannot hold encoder plus DiT, moot on
unified memory); FP8 weights (memory only, 8.4-8.8 s against 7.2 s bf16, 26.1 dB PSNR all
layers, 29.7 with `img_mlp` kept bf16); FP8 prefix KV (`fp8` 34.9 dB, `fp8_v` 40.9 dB,
storage only, material near their ~24.6k-token limit where a branch's cache is 12.9 GB).

**The VAE-at-2K warning.** They tile by default (`--vae-use-tiling` in CI; 512 px tiles
at 384 stride, linear blends on the 25% overlap; on OOM halve the tile down to 128 px and
retry) and note "the default-tile banding limitation at 2048px+"; diffusers' 256/192
default read 39.6 dB against 47.7 for 512/384 versus untiled. Their decoder is a Wan-style
conv3d with feature caches where diffusers' 2.1 class is the 2-D squeeze this doc ports,
and no source states the untiled 2048x2048 peak. So the untiled native-size decode is a
risk to MEASURE before it is served, and 512/384 tiling with blended overlaps is the
fallback sketch. Their unverified flags carry over: the scheduler shift values,
`causal_condition` and the VAE constants were code defaults on their side too, and no
vllm-omni number exists at 2048x2048 or 40 steps.

## Cost estimate, to be measured

Per image token per block, projections and FFN only, two FLOPs per multiply-add:

- Z-Image: `4 x 3840^2 + 3 x 3840 x 10240 = 176.9 M` params per block; 32 blocks touch
  an image token (30 main + 2 noise refiner; the 2 context refiners touch caption tokens
  only), so 11.3 GFLOP per token, or 12.0 at the 34 `tests/zimage_microbench.rs` counts.
- Qwen-Image 2.1: `4 x 4096^2 + 3 x 4096 x 12288 = 218.1 M` per block, 32 blocks,
  14.0 GFLOP per token.

So 1.16-1.23x per token on the same 4096 image tokens at 1024x1024, and the text tokens
drop out of the per-step cost thanks to the cache (Z-Image carries a 32-token padded
caption every step; 2.1 none after step 0). Attention adds `4 x S x 4096` per token per
block, 13% of the linears at S = 4096 and 61% at 16384.

Against the Z-Image figure in [perf-state.md](perf-state.md) "Z-Image-Turbo, a time per
image and not a tok/s target", 1.35 s first step rising to 2.0-2.1 s by step 8 at
1024x1024 under a hardware clock limiter in automatic power mode: a 2.1 step at
1024x1024 should read about 1.6 s cold rising to 2.3-2.6 s at the plateau, so 40 steps
of the order of 90-105 s and 25 steps (ComfyUI's template) about 60 s. At native
2048x2048 the linears are 4x and attention 16x per step, together about 5.6x: a step of
the order of 10-14 s and a 40-step image 7-10 minutes at the plateau, plus the unpriced
decode. **Derived from the Z-Image measurement, not measured**, assuming the same
envelope; the parity bar is what gates shipping, not any of them.

Editing, derived the same way. With k references at about 1 MP each the DiT prefix grows
by k x 4096 clean latent tokens plus the VL text tokens (the ~1026 encoder tokens per
image never reach the DiT). Step 0 runs (1 + k) x 4096 image rows plus the text through
the 32 blocks, roughly (1 + k) times a T2I step 0; every later step is a T2I step (4096
target rows through the linears) plus attention over a (1 + k) x longer K/V, about +13%
per reference at the 13% figure above, so two references read about 1.25x a T2I step and
ten about 2.3x. The encoder pass adds the ViT, 0.85 TFLOP per image at ~1024 tokens, and
the 8B text stack's prefill over ~1026 more tokens per image. The cache is the memory
line: a 1 MP reference is 4096 x 32 x 2 x 4096 x 2 B = 2.15 GB of f16 K/V, ten of them
21.5 GB, which is what the two-base-pointer read would stop copying per block per step.

## Verification plan

Mirror the Z-Image gate. `tests/zimage_parity.rs` gates the step-0 velocity at cosine
>= 0.998 and mean relative error <= 0.04 (:51-53), bars set from the reference's OWN
bf16-against-fp32 spread carried in `meta.json`, with two deliberately wrong graphs
asserted outside the bar every run; the VAE alone at PSNR >= 60 dB (:58); the final
latent and image PSNR after the full schedule REPORTED, not gated (decisions.md "An
agreement bar is bracketed from both sides or it is not a bar"). The fixture
`tests/fixtures/zimage-transformer/<case>/` holds BOTH inputs (noise, and caption
features rounded once to bf16) so the gate grades the transformer and not the encoder,
plus `velocity0`, `latents-final`, `image` and `meta.json`, about 1.5 MB;
`scripts/zimage-ref-dump.py` writes it under `uv` in a throwaway venv, the one kind of
Python the repo allows, by hand and never in CI ([AGENTS.md](../AGENTS.md), "The Python
exception"); `scripts/qwen-image-ref-dump.py` inherits exactly that status. Runbook
shape: [parity.md](parity.md) "Running the three gates".

The sibling pins diffusers at or after `6256aa7` and `transformers>=5.17` and installs the
pipeline's pre-norm hook rather than trusting `hidden_states[-1]`. Stages:

1. **Encoder.** Rendered string byte-equal, ids, drop index, and the pre-norm hidden state
   `[T_kept, 4096]` in fp32 against `encode` at depth 36 pre-norm, on the Z-Image Stage-2
   prompt set. Bracket: the NORMED state, the transformers-5 default.
2. **Transformer step 0.** Velocity from fixed noise and fixed caption features at sigma
   1.0, cosine and mean relative error, bars from the reference's own spread. Brackets:
   full bidirectional attention over the joint sequence (the 1.0/Z-Image pattern) and
   every token modulated from the real `t` (no t=0 row). No-cache is NOT a bracket.
3. **VAE decode alone.** The reference's final latent through xwen's decoder against the
   reference PNG's RGB, PSNR >= 60 dB, both sides f32.
4. **Full 40-step image** under `use_kv_cache=True` on the reference, reported.
5. **Editing** (Phase 5): the encoder with the vision tower run, an edit fixture's step 0
   and the full image, with brackets, as Phase 5's gate lists them.

Two bisect arms per stage, as Z-Image has, so a bar can be attributed.

## Phases

Each phase is an arc with its own record, closed by its gate.
- **Phase 0, references and pins.** Entry: the registry entry sketch and the `uv` env.
  Prerequisites: the license decision, egress to `huggingface.co`, one fetch. Work: list
  the HF tree, read shard hashes and safetensors dtypes, confirm the four mirrored
  configs, pin diffusers and transformers, run the four dumps at 1024x1024 with a fixed
  seed. Gate: fixtures committed with `meta.json` carrying the reference's own spread.
- **Phase 1, the encoder.** Entry: `src/qwen3/config.rs`, `src/qwen3/safetensors.rs`,
  `EncoderSpec` in `src/hub.rs`, `src/zimage/conditioning.rs`, `xwen encode-text`. Work:
  the seven-item extension list, the MRoPE handling DESIGNED for the real three-axis case
  (a `RopeSpec` with sections, the table code taking three id rows per token) though only
  the degenerate text case runs here. Gate: Stage 1 at the Z-Image bars (cosine
  0.9999-class, the same graph), the normed bracket outside it, and the tokenizer
  byte-compare test extended to the new file.
- **Phase 2, the transformer and scheduler.** Entry: new `src/qwen_image/`
  (`transformer.rs`, `pipeline.rs`, the scheduler arm), the `CheckpointSource` diffusion
  arm, `xwen image --model qwen-image-2.1`. Prerequisite: Phase 1 for a real caption; the
  fixture works without it. Work: CFG-free T2I with the prefix K/V kept across steps,
  dynamic shift, the size rule, `--latents`/`--cap-feats`/`--dump` as Z-Image has
  (pipeline.rs:353-388, :396-421, `Rendered`); the joint sequence takes a LIST of image
  blocks with block ids and per-block rope frames from the start, T2I the one-block case,
  so Phase 5 adds rows and not a second assembly. Gate: Stage 2 at its bars, both
  brackets outside; Stage 4 reported. The VAE is candle's arm here.
- **Phase 3, the VAE decoder and encoder.** Entry: `src/qwen_image/vae.rs`, then
  `src/ops/` for the per-pixel RMS fold and the direct-conv fusion. Work: the decoder on
  the candle arm first, then the direct arm priced by microbench (it will pay at
  2048x2048); the ENCODER (RGBA in, `conv_in` 4 -> 96 on candle, posterior mean, the
  `(z - mean) / std` normalisation) built alongside because editing needs it. Gate: 60 dB
  on both decoder arms; a reference's encode-then-decode round trip reported.
- **Phase 4, serve, memory, figures, docs.** Entry: `src/serve/images.rs`,
  `src/memory.rs`, `docs/perf-state.md`, `docs/qwen-image.md`,
  `docs/decisions/qwen-image.md`, `docs/log.md`, README. Work: the model lookup on the
  route and in the envelope, per-model peaks measured at 1024x1024 and 2048x2048 (the
  untiled decode among them), `negative_prompt` plus `true_cfg_scale` validated per
  model, a sibling perf-state section quoting a step as a range with its ramp and `pmset`
  line (decisions.md "A Z-Image step is quoted at steady state"). Gate: docs-check green,
  the figure in place.
- **Phase 5, editing in the engine.** Entry: `src/qwen_image/vision.rs` (preprocessor,
  ViT, mergers), `src/qwen3/` (vision position ids, pad-token scatter, DeepStack adds),
  `src/qwen_image/conditioning.rs` (the tag renderer), `src/qwen_image/pipeline.rs`
  (condition latents), `src/serve/image_control.rs`, `src/serve/images.rs`,
  `src/bin/xwen/main.rs`, `src/zimage/inputs.rs` (RGBA-aware `prepare_image`).
  Prerequisites: Phases 1-3 shipped, the `uv` env extended with the vision processor, the
  shipped `preprocessor_config.json` read. Work: the editing section end to end, then
  `xwen image --reference`, `references` on the native route, repeated `image[]` on
  multipart. Gate: the encoder on a reference-image prompt (rendered string byte-equal,
  pad count per image, pre-norm hidden state with the vision tower run, at the Stage-1
  bars; brackets: MRoPE not applied to the image run, DeepStack skipped); step 0 on an
  edit fixture holding noise, caption features AND the clean condition latents of two
  references, at the Stage-2 bars (brackets: condition rows modulated from the real `t`,
  one block id for all images); the full image on an official README example, PSNR
  reported.
- **Phase 6, the GUI (Image Studio).** Entry: `image-studio/src/domain.ts`,
  `components/SettingsPanel.tsx`, `useImageDrop.ts`, `batchDraft.ts`,
  `src-tauri/src/models.rs`, `state.rs`, `storage.rs`, `e2e/`. Prerequisites: Phase 5's
  HTTP contract frozen and the images models listing landed. Work: item (d) of the GUI
  section. Gate: `bun test src` and Playwright green over the fixture bridge with a
  two-reference edit fixture, one edit rendered end to end against a local serve, the
  record and the PRD "Clients" section revised. No correctness bar.

**Later, and explicitly not in this plan**: true CFG as a SERVED option beyond the seam
named in Phase 4 (cheap once two prefix caches exist, the vLLM-Omni section's item 5, but
unpriced and not what the model card samples with); LoRA (`lora.rs` is generic mechanism
with Z-Image target names); RGBA output with the model card's transparency prompt wrapper;
GGUF-quantized transformer weights (leejet's Q4_K/Q8_0 exist; the matmul path is bf16,
decisions.md "The transformer runs bf16 end to end"); the chat sidebar's tool for edits
(`queue_txt2img` stays txt2img-only,
[records/image-studio.md](records/image-studio.md#assistant-chat-and-queue-tool)).

## Open questions and unverified facts

- The HF file tree, shard count, byte sizes and on-disk dtypes of `transformer/` and
  `vae/`; the ~33.1 GB repo total; Comfy repack and leejet GGUF sizes (all snippets or
  third-party, though the two bf16 sizes agree with the computed parameter counts).
- The literal `transformer/`, `vae/` and `scheduler/` configs (mirrors agreeing with the
  class defaults); the contents of `processor/tokenizer_config.json`,
  `text_encoder/generation_config.json` and `model.safetensors.index.json`.
- Whether the `text_encoder/` tensor values equal standalone `Qwen/Qwen3-VL-8B-Instruct`
  (files differ, re-saved), and Instruct against Thinking base.
- The non-Base `Qwen/Qwen3-4B` tokenizer sha and the regex inside the 11.4 MB file, hence
  the `VocabFamily::Qwen3` claim, pending the byte-compare test.
- The exact drop index on the shipped file (14 on a re-serialised copy); compute it.
- The MRoPE degeneracy to plain NEoX for text (inferred); confirm numerically in Stage 1.
- The `RMS_norm` form (L2 per pixel over channels x sqrt(C), 1e-12 clamp); the VAE-alone
  gate confirms it.
- The size rule for T2I: multiples of 32 px per the reference's floor, against the 16 px
  the latent stride alone suggests; the doc takes 32.
- The rope extent: 2048 latent tokens per side by the table's negative rows; confirm.
- The license decision (non-commercial), before Phase 0; and whether a 1024x1024 default
  is served where the native size is 2048x2048.
- The doc drift at docs/decisions/zimage.md:36-38 (four `Format::Diffusion` fields
  described, one in the code).
- How `EncoderSpec` expresses "depth 36, pre-norm" against a convention that norms there.
- The absence of an official distilled or fp8 variant, and whether 2.0 weights were ever
  public (snippets only).
- Whether `<imageN>` is plain text in the shipped `tokenizer.json` (every consumer read
  treats it as text; the file was not read).
- The README's 2048x2048 default against the pipeline's `output_resolution` 1024, which
  sets both the output area and the reference resize; the doc follows the pipeline, and
  the last-reference (diffusers) against first-reference (ComfyUI) size rule with it.
- The shipped `preprocessor_config.json`: `min_pixels`/`max_pixels` (the class default is
  below 1024^2 and would break the slot count), mean/std (0.5 per ComfyUI, CLIP per the
  class default), resample; and the ViT rope theta and `hidden_act`, all read at Phase 0.
- Whether `/v1/models` should list image models, or a `GET /v1/images/models` beside it;
  the doc chooses the latter, for the chat clients' sake.
- The untiled 2048x2048 VAE decode peak (no source states it; vLLM-Omni tiles by default);
  measured at Phase 4 before the native size is served.
- The two-base-pointer K/V read in `flash_t.metal` that would remove the per-step
  `[prefix ; target]` copy: whether the copy costs anything on Metal is unpriced.

## Sources

External, all read 2026-09-21: the QwenLM README and LICENSE
(https://raw.githubusercontent.com/QwenLM/Qwen-Image-2.1/main/README.md, `.../LICENSE`);
diffusers PR #14804 (https://github.com/huggingface/diffusers/pull/14804) and on `main`
at 6256aa7 `transformer_qwenimage21.py`, `pipeline_qwenimage21.py`,
`autoencoder_kl_qwenimage21.py`, `scheduling_flow_match_euler_discrete.py`; transformers
`main` `modeling_qwen3_vl.py` and https://github.com/huggingface/transformers/pull/48087;
ComfyUI `comfy/text_encoders/qwen_image_21.py`, `comfy/supported_models.py`,
`comfy/model_detection.py`, `comfy_extras/nodes_qwen.py`; the sd.cpp doc
(https://raw.githubusercontent.com/leejet/stable-diffusion.cpp/master/docs/qwen_image_2.1.md);
the candle models tree. Mirrored configs and manifests: `Blizaine/Maestro@5efd686`,
`wee-todd/WeeTodd-Studio`, `Enntity/lloom`, `ostris/ai-toolkit`, `rcarmo/go-pherence`,
`yairpatch/flyweight`.

For the editing section, also read 2026-09-21 from `raw.githubusercontent.com`: diffusers
`docs/source/en/api/pipelines/qwenimage21.md`, the `qwenimage21` pipeline and transformer
tests, `image_processor.py`; transformers `qwen3_vl/modeling_qwen3_vl.py`,
`configuration_qwen3_vl.py`, `processing_qwen3_vl.py`,
`qwen2_vl/image_processing_qwen2_vl.py`, `vision_utils.py`; ComfyUI
`comfy_extras/nodes_qwen.py`, `comfy/text_encoders/qwen_image21.py`, `qwen3vl.py`,
`qwen35.py`, `qwen_vl.py`, `llama.py`, `comfy/ldm/qwen_image21/model.py`, `comfy/sd.py`,
`comfy/supported_models.py`; `Comfy-Org/workflow_templates`
`image_qwen_image_2_1_image_edit.json`; vLLM `vllm/model_executor/models/qwen3_vl.py`.
For the performance section: vLLM-Omni PR
https://github.com/vllm-project/vllm-omni/pull/7759 at
`3ea9e605011fc2a0bfa9f3e8531286fb683068dd` (`vllm_omni/diffusion/models/qwen_image_21/`,
`diffusion/attention/backends/`, `diffusion/sched/`, `recipes/Qwen/Qwen-Image-2.1.md`,
`docs/user_guide/quantization/fp8.md`) and the PR pages #7769, #7787, #7790, #7797.

Session reports, scratch and not committed, under
`/tmp/claude-0/-home-user-xwen/d2a8c7b2-a5d2-576c-977b-407d98011c43/scratchpad/`:
`qwen-image-release.md`, `qwen-image-diffusers-math.md` (the math, the porter's trap
list, the sigma grids), `qwen-image-text-encoder.md`, `xwen-diffusion-seams.md` (every
file:line in this doc that points into the tree), `qwen-image-edit-path.md` (the editing
path and the vision tower), `xwen-gui-and-edit-surfaces.md` (the GUI, CLI and HTTP edit
surfaces with file:line), `vllm-omni-qwen-image.md` (the vLLM-Omni techniques); raw copies
of every external file read sit under `scratchpad/src/` beside them.
