# Qwen-Image 2.1

One topic of [docs/decisions.md](../decisions.md), the index of decisions by topic: the
second diffusion pipeline and everything decided about it. Dated paragraphs, newest
additions appended. The architecture and the traps are in
[docs/qwen-image.md](../qwen-image.md), the arc is
[docs/records/qwen-image-t2i.md](../records/qwen-image-t2i.md), the research that came
first is [docs/qwen-image-2.1-plan.md](../qwen-image-2.1-plan.md), and the scope
amendment that lets image transformers in is in [scope.md](scope.md).

**Qwen-Image 2.1 is a top-level module written from diffusers, not vendored and not
under `src/zimage/`.** candle has no module for this model in any version, so there is
nothing to vendor, and `src/zimage/` is a vendored-and-corrected candle module for one
model with a never-resync rule that a second model would blur. `src/qwen_image/` is
ported from diffusers at `6256aa7` with the reference sources read line by line, and the
plan's architecture table was treated as a checklist of traps rather than as the spec,
which paid off: the reference contradicted the plan in nine places
([the record](../records/qwen-image-t2i.md), "Where the plan was wrong"). It reuses the
Z-Image seams as they are: `zimage::linear::Projection`, `ops::flash_attn_tensor`,
`ops::rope_pair`, `ops::gated_residual`, `ops::silu_mul`, `seeded_noise`, `encode_png`.
No kernel was written for it (2026-09-21).

**The text encoder opens through the dense Qwen3 loader, extended, and there is no
`src/qwen3vl/`.** The Qwen3-VL-8B text tower is the graph `src/qwen3/` already runs and
grades in every form that could have gone the other way, and its tokenizer is
byte-identical to Qwen3-4B's (sha256 `aeb13307...`), so a second module would have been a
second copy of the one graded encoder, which the repo refused once for Z-Image
(zimage.md, "Text conditioning comes from `XwenModel::encode`"). What was extended: the
nested `text_config`, the `model.language_model.` prefix with `model.visual.*` and the
untied head left out of routing, `rope_scaling` accepted only as interleaved MRoPE with
sections summing to `head_dim / 2`, and a pre-norm tap. An untied set is refused on every
language path and accepted by `load_encoder` alone. The vision tower stays unloaded and
Phase 5 loads it as a module beside the text stack, not as a second encoder. Evidence
that the reuse is sound: minimum cosine 0.99997 against the fp32 reference over 12
prompts, on the plain NEoX table (2026-09-21).

**"Depth 36, before the final norm" is a field on the spec, and one helper reads it.**
The repo's index convention applies the final norm at index `n_layer`, and the reference
conditions on the residual before it, which transformers 5.17 no longer hands back from
`hidden_states[-1]` without a hook. `EncoderSpec::final_norm` and
`qwen3::HiddenTap { depth, final_norm }` say it, `XwenModel::encode_spec` is the single
entry for spec-driven callers (`encode-text`, `xwen image`, the serve images engine), and
bare `encode` keeps its meaning for the tests that index explicitly. The review that
asked for the helper found two call sites where the field could never have been honoured.
The normed state reads minimum cosine 0.62 against the right one, so the mistake is loud
in the gate and silent in an image (2026-09-21).

**Literal special-token text in a prompt stays plain text, which is a deliberate
difference from the reference.** The HF processor parses `<|image_pad|>` or `<think>`
typed into the user's text as the special token. xwen's `encode_prompt` rule does not,
and it is kept here because once editing lands a prompt that could mint image slots
could change the slot count the transformer asserts on. It affects one fixture prompt,
marked `literal_specials`, where the test asserts inequality with the processor's ids and
equality everywhere else, and the transformer dump refuses that prompt (2026-09-21).

**A prompt past 4096 tokens is refused, not truncated.** The reference pipeline has no
`max_sequence_length`. The cap is ours, generous, and a refusal because the template
ends on the open assistant turn, which is the part a truncation would cut (2026-09-21).

**The scheduler is its own file, not a second arm inside the vendored one.** The plan
said to add a dynamic-shift arm to `src/zimage/scheduler.rs` and keep the static one
untouched. `src/qwen_image/scheduler.rs` does it as a new file instead, about 350 lines,
so the vendored file stays as it was corrected and the two schedulers share nothing they
could disagree about; the cost is a second `step`, which is one line of arithmetic. An
outside review reproduced the 40-step grid at 4096 tokens independently and matched every
f32 entry bitwise (2026-09-21).

**Block-causal attention runs as segments on the existing kernels, with no dense mask on
the shipped arm.** `(q >= kv) OR same_image_block` decomposes into one call per prefix
segment over keys `[0, end)`, causal for text alone, and one call for the target over
every key, which is what diffusers' non-flex processor does. `ops::flash_attn_tensor`
takes independent query and key extents, so the target call is the shipped kernel
unedited. The dense-mask arm exists as the off-Metal path and the bisect arm, and the CPU
tests hold the two equal to 1e-4 with zero and with two reference blocks. The sequence is
an ordered segment list because the reference puts reference-image rows INSIDE the text
stream, not after it (2026-09-21).

**The prefix K/V cache is on by default, and running without it is a bisect arm, not a
wrong-graph bracket.** The reference sample is produced under `use_kv_cache=True`, and
the text and reference rows are modulated from a fixed `t = 0` row and attend only to
themselves, so they are the same at every step. Keeping their K and V means a later step
runs the target rows alone. `XWEN_QWEN_IMAGE_CACHE=off` is the same math at a different
rounding, so it cannot bracket a bar: the brackets are the two graphs that are actually
wrong (2026-09-21).

**The parity bars are Z-Image's, and both wrong graphs fall outside them.** Step-0
velocity at cosine 0.998 and mean relative error 0.04, the VAE alone at 60 dB, the
40-step image reported and not gated. The reference's own bf16 arm reads 0.999948 and
0.0075 against its fp32 arm, inside the bar; xwen reads 1.000000 and 0.0006; fully
bidirectional attention reads 0.9825 and text modulated from the real `t` reads 0.9874,
both outside. The second bracket was the uncertain one, the real timestep and the `t = 0`
row being close at sigma 1.0, and it separates. The VAE alone reads 91.07 dB and the
40-step image 57.30 dB, where the reference's bf16 arm gets 35.58 (2026-09-21).

**The encoder gate's bars are set from the reference's own spread, and kept row 0 has a
bar of its own.** The bar inherited from Z-Image, cosine 0.9999 and relative error 0.01
on every row, failed 1 to 4 rows a prompt on a graph that was evidently right. Kept row 0
is the user turn's `<|im_start|>`, the same computation in every prompt, and it carries a
massive activation inside the stack: residual norm about 9390 at layers 24 to 34,
cancelled back to about 870 by the last layers, so its value is a small difference of
large numbers. xwen reads relative error 0.0254 there and torch's own bf16 arm 1.58. It
gets cosine 0.9999 and relative error 0.05, twice xwen's figure, 30 times inside torch
bf16 and 20 times inside the normed wrong graph. Past row 0 the bar is cosine 0.9999 and
relative error 0.03, under the MEDIAN row of torch's bf16 arm (0.034) and 13 times under
the wrong graph's best row (0.404), and 99% of rows must still hold 0.01, which xwen does
at 99.67%. The six rows that miss 0.01 are function words and the literal `<think>`, and
they are the rows torch's bf16 arm is worst on too. A non-ignored test asserts each
constant against `reference.json`, so a bar cannot leave its bracket. The
massive-activation figures came from a one-off torch probe and are not reproduced by the
script (2026-09-21).

**The image is returned RGBA only when at least 10 pixels have alpha at most 8.** The
model always decodes four channels. An ordinary prompt's alpha plane is almost opaque
and not exactly: the diffusers fp32 reference at 512x512 has 217,245 pixels at 255,
44,891 at 254, 7 at 253 and 1 at 252, so "any pixel that is not opaque" would return
every image slightly translucent, to clients that assume opaque PNGs. With the model
card's transparency prompt the background sits at alpha 1 and not 0, on diffusers
(120,507 pixels at exactly 1, 1,619 at 0) and on xwen (158,908 and 3,895), which is why
the cutoff is 8: counting exact zeros would have fired on about 1% of the clear area. The
count of 10 guards a stray pixel and needs no tuning by resolution. Ordinary prompts read
a minimum alpha of 252 at 512x512, 250 at 1024x1024 and 230 at 8 steps, with no clear
pixel at any of them. The accepted miss is an image that is ONLY semi-transparent, smoke
or glass with no fully clear pixel, which comes back RGB; reopen with a request field
that forces RGBA when someone needs that, which costs nothing to add beside the rule
(2026-09-21).

**The u8 rounding of exact ties is left as it is.** `zimage::sampling::postprocess_image`
rounds half away from zero where diffusers' NumPy rounds half to even, one level on an
exact `.5` after the scale to 255. It is a shared helper, Z-Image's figures already carry
it, and no bar can move on it: the 40-step image reads 57.30 dB and the VAE alone 91.07.
Reopen if an image PSNR figure is ever shown to be limited by ties (2026-09-21).

**Memory admission for this model is a line through two measured peaks, and the 1 MP cap
stays.** The first 1024x1024 render peaked at 71 to 73 GiB during the VAE decode against
a guessed 46 GiB, which is an admission that lies, and the coordinator plans from
declared peaks (records/memory-safety.md). With no sync inside the decode candle's pool
kept every conv intermediate alive; syncing between convolutions, which is what evicts
the pool on Metal, takes it to 55 to 56 GiB at 1024x1024 and 25 GiB at 512x512 with
byte-identical PNGs. `peak_bytes` is 17 GiB plus 48 GiB per megapixel, the fit through
those two points (14.7 plus 41.3) with 15% on top, 65 GiB and 29 GiB, and a unit test
holds it above the measured constants. What is left of the decode peak is one layer's
im2col column buffer, 10.9 GB for the 288-channel conv, against a step phase of 27 GiB:
that is what prices the direct-conv arm or tiling. The 1,048,576-pixel cap is kept
because the native 2048x2048 has no measured peak, and its message says so. Whether the
drains slowed the decode by more than 5% could not be told apart from noise, the machine
being in low power mode for those runs (2026-09-21).

**`xwen image` fetches an uncached model with a notice, and `auto_fetch` false governs
serve alone.** The brief for the wiring asked for an error naming `xwen fetch`. The CLI's
existing convention for Z-Image is to announce the size and fetch, and a second rule for
a second image model would be the surprise. What did change is what is fetched and when:
`--cap-feats` resolves only the files the pipeline loads, where it used to pull 17.5 GB
of encoder shards it never opened (Z-Image had the same flaw and got the same fix), and
the tokenizer alone is resolved before the weights so an overlong prompt is refused
first (2026-09-21).
