# Z-Image-Turbo

One topic of [docs/decisions.md](../decisions.md), the index of decisions by topic; the
text-to-image pipeline and everything decided about it. Dated paragraphs, newest
additions appended. The architecture and the traps are in
[docs/zimage.md](../zimage.md), the arc is
[docs/records/zimage-pipeline.md](../records/zimage-pipeline.md), and the scope
amendment that let any of it in is in [scope.md](scope.md).

**candle's `z_image` module is vendored into `src/zimage/`, not depended on in place and
not written from scratch.** Three reasons, in order. It was already correct on every
trap the architecture research had identified as a silent-garbage risk: interleaved-pair
rope rather than NEoX, image-first sequence order, `scale + 1` with `tanh` gates and no
shift term, cos-before-sin in the timestep embedding, both learned pad tokens loaded.
That is a real head start on the exact places a reading can go wrong. Second, it is also
wrong in four places that only execution or a close reading finds (the corrections are
listed in [docs/zimage.md](../zimage.md)), and fixing upstream code we do not control,
in a dependency pinned by revision, is not a workflow: we need to edit it, instrument
it, and put kill switches in it. Third, this module is the base for hardware-specific
speed work, which is the whole point of the repo, and vendoring is how every other
kernel-level thing here is held. The provenance line is candle rev 21cca0b, PR #3261
(2026-09-07).

**Text conditioning comes from `XwenModel::encode`, never from candle's own text
encoder.** candle's `z_image/text_encoder.rs` is a second, independent implementation of
the same Qwen3-4B read, and it arrives at the same conclusion about the hidden-state
index by a different route, which is useful corroboration and nothing more. This repo's
encoder is the one with a reference gate on it: minimum cosine 0.99999449 and maximum
relative error 0.00388 against a torch fp32 dump, about ten times closer to fp32 than
the reference pipeline's own bf16 execution ([docs/zimage.md](../zimage.md)). Using an
unverified adapter beside a verified encoder would throw the only measured thing in the
conditioning path away, and it would leave the pipeline reading a hidden state nobody
had graded. `src/zimage/` therefore has no encoder of its own; it takes a `[T, 2560]`
caption tensor and the pipeline gets it from `encode` (2026-09-07).

**The pipeline is a new registry entry naming every file in the repo, and the
encode-only entry stays beside it.** `Model::ZImageTurbo` carries a new
`Format::Diffusion { text_encoder, transformer_config, vae_config, scheduler_config }`
rather than extending `Format::SafeTensors`, because `is_safetensors()` means "a Qwen3
set the Qwen3 loader opens" and `identify_cached_dir` iterates exactly those: a
pipeline-shaped safetensors entry would have made the snapshot root identify as a
language model, which an existing test pins as identifying as nothing. Exactly one of
`is_gguf()`, `is_safetensors()` and `is_diffusion()` is true per entry, and that is
tested. The entry lists all fifteen files, the encoder's five among them, so one fetch
leaves nothing to download; `model_index.json` is listed first so the resolved path's
parent is the snapshot root, which is what `xwen image` opens. The encoder entry is not
folded into it: `encode-text` is a real surface with its own reference fixtures, and the
two entries want different `not_servable_reason()` sentences. What moved is the alias.
`zimage-turbo` now names the pipeline and the encoder became
`zimage-turbo-encoder` (2026-09-07).

**Weights load through candle's `VarBuilder`, fp32 on disk cast to bf16 at load, and
there is no third `CheckpointSource` arm this arc.** `CheckpointSource` is the repo's one
seam for opening a checkpoint and every language-model consumer routes through it, so a
diffusion arm belongs there in principle. It is not built yet because there is exactly
one consumer, `ZImagePipeline::load`, and a seam with one caller is a guess about the
second one. The reopen condition is literal: when a second consumer needs the
transformer or the VAE opened (the serve images route is the likely one), build the arm
then, with two call sites to shape it. The dtype decision is separate and is not
deferred: the shipped shards are F32, 24.6 GB, and the reference executes bf16, so the
cast happens at load, one tensor at a time, and the resident transformer is 12.3 GB
(2026-09-07).

**The transformer runs bf16 end to end. fp16 is REFUTED, and fp8 and Q8_0 are not
taken.** bf16 is what both reference implementations execute and it is the arithmetic
the correctness bars will be read against; whether F32 activations against bf16 weights
would buy anything here, the way they demonstrably do in the encoder, is a question for
the step-0 parity gap and not for a guess before the gap exists. fp16 is refuted with
evidence rather than deferred: Z-Image's activations exceed fp16's 65504 ceiling, and
upstream `Tongyi-MAI/Z-Image` issue #14 reports pure black images from NaN latents in
`torch.float16` while bf16 and fp32 are fine, corroborated on the training side by
`kohya-ss/musubi-tuner` issue #897. Anything that introduces fp16 storage or fp16
accumulation into this graph is a correctness risk, and the FFN down-projection is where
it blows up. fp8 and Q8_0 are declined for a different reason, which is that they cannot
help: a step is roughly 62 TFLOP against 12.3 GB of weight traffic, an arithmetic
intensity near 5100 FLOP/byte against a ridge point of 24 to 114, so reading the weights
is about 20 ms of a multi-second step and halving the bytes saves about 10 ms. On top of
that candle has no fp8 dtype at all and its Metal quantized GEMMs are dequant-in-kernel
`simdgroup_matrix` kernels that bypass the tensor path a bf16 matmul reaches, so a
quantized arm here would likely be slower at the same FLOPs. Quantization is a footprint
lever on this graph and nothing else. Reopen on either of two conditions: a W8A8 kernel
that reaches the M5 neural accelerators, which raises the ceiling instead of lowering
the byte count and is the only precision lever that could move a step, or footprint
pressure that makes 12.3 GB the problem (2026-09-07).

**Verification is a torch dump with an injected latent, gated at the step-0 velocity
field and reported at the final image.** The pipeline's noise draw cannot be reproduced
from Rust (torch draws on the device, CUDA Philox or CPU MT19937, and both quick-starts
seed a device generator), so a fixed `[1, 16, 128, 128]` fp32 latent is dumped once and
fed to both sides; from there the whole run is deterministic modulo kernel arithmetic.
**Stage 3** grades the step-0 velocity field, cosine and relative error, and it is the
gate: one forward exercises all 34 blocks, the 3-axis rope, every modulation and the
final layer, which makes it the highest-value single tap available. **Stage 4** grades
the final image by PSNR against the official pipeline on the same latent, and is
reported rather than gated, because eight Euler steps compound arithmetic differences
and a bitwise bar on a PNG is a bar on nothing. The bars themselves are deliberately not
set here: they get decided when the dump exists and its own spread is known, which is
exactly how the dense Qwen3 Stage 1 bar had to be decided
(decisions.md "The Stage 1 oracle for `qwen3` is per-position logits"). The dump extends
`scripts/zimage-ref-dump.py` under the standing Python exception rather than becoming a
second Python entry point, and the official repo's MPS branch is what makes it runnable
on this machine. The requirement on it is that it be fast: an oracle that takes hours
does not get re-run, and an oracle that does not get re-run stops being one (2026-09-07).

**First image before first bar.** The order was deliberate and it is worth recording as
a choice rather than as an accident: get a coherent 1024x1024 PNG out of the vendored
module end to end, then build the reference dump, then do performance work. The argument
is that a whole-pipeline failure is cheap to see and expensive to bisect, and that a
graded intermediate proves nothing about the parts nobody wired up. It paid: four
corrections against candle were found by getting the thing to run and reading the
reference beside it, and none of them needed a dump. It also means the module is
unverified numerically at the end of Arc A, which is stated in the record rather than
implied by silence (2026-09-07).

**1024 square first, and the image pad-token path is refused rather than half
implemented.** `check_size` accepts a width and height only when both are positive
multiples of 16 and the image token count `(w/16) * (h/16)` is a multiple of 32; it
refuses everything else with the reason, naming the token count when that is the half
that failed. 1024x1024, 1024x768, 768x1024, 512x512 and 1536x1024 pass; 1000x1000 fails
on the multiple of 16 and 528x528 fails at 1089 tokens. The transformer's real rule is
looser than that, because `x_pad_token` exists precisely so a ragged image grid can be
padded up to a multiple of 32 the way the caption is. Implementing it means pad rows at
rope position (0,0,0) that participate in attention as ordinary keys and queries, and
that is a piece of math with no oracle behind it yet, so it would be untested silent
garbage on a size nobody asked for. The refusal is the honest version and it is a ledger
item, not a hidden limit (2026-09-07).

**CLI first, then serve as OpenAI `POST /v1/images/generations`, plus a ComfyUI node of
our own.** The user is waiting to drive this from ComfyUI, which decides the wire shape:
the OpenAI images API is the one shape where a ComfyUI node exists today that talks to
an arbitrary `http://localhost:PORT` and hands back an `IMAGE` tensor with no glue, and
it is where every other OpenAI-compatible local image server converged. Extension fields
`negative_prompt`, `num_inference_steps`, `guidance_scale` and `seed`/`rng_seed` ride as
top-level keys, which is the convention four independent implementations already agree
on; `quality`, `style`, `n` and the rest of the OpenAI surface are accepted and ignored
in the usual way. A node of our own is part of the decision rather than a fallback: no
server-side shape drags clients along, the best-fitting existing node has one author,
and 120 lines of Python we control is cheaper than contorting the endpoint to fit clients
that barely exist. The endpoint stays the thing the node calls, so it keeps being useful
to everything else. One requirement that is easy to miss and belongs in the design from
the start: an image model must participate in the existing idle-unload behaviour, because
12.3 GB resident on a server that is answering language requests is not acceptable
(2026-09-07).

Amended the same day, after the user refused to install a community node pack: the node
of our own is no longer part of the decision, it is conditional. ComfyUI's stock OpenAI
image nodes POST to the relative path `/proxy/openai/images/generations` against the
`--comfy-api-base` flag, with no auth header when there is no comfy.org token, so the
route must ALSO answer at that path, speak the OpenAI error envelope, and never return
401, 402, 409 or 429 (the node rewrites those into comfy.org login and credit prompts).
That is the zero-install path and it is the deliverable. Its limits are the stock node's:
seed never reaches the wire, size is a dropdown of 1024x1024, 1024x1536, 1536x1024 and
auto, no steps or negative prompt, and the flag redirects every partner node at once. The
mechanism is confirmed in ComfyUI's source and unattested in the wild, so the route ships
with a curl against the proxy path and is proven from the laptop before anything else is
built. A node pack of our own (three nodes with real CONDITIONING and LATENT sockets, so
an xwen latent can feed local nodes and a local latent can come back for img2img) is
built only when the user names a composition that needs it: img2img, inpainting, LoRA or
an upscale chain. Making xwen a ComfyUI backend that speaks the frontend's own protocol
is refuted as an approach: a large arc reimplementing a moving, loosely documented API for
a fixed node set (2026-09-07).

**The encoder and the transformer are both resident for the whole run.** diffusers offers
the other arrangement, `model_cpu_offload_seq = "text_encoder->transformer->vae"`, and it
exists because 8 GB plus 12.3 GB is a real problem on a 16 GB card. It is not one here:
this machine holds both, `xwen image` encodes once and then denoises eight times, and
dropping the encoder between them would buy footprint we are not short of at the price of
a reload on the next prompt. The serve route will want the opposite of offloading anyway,
since sharing one loaded Qwen3-4B between the language surfaces and the image surface is
one of the two reasons this endpoint is worth having at all (2026-09-07).

**The sigma grid follows diffusers, not the official repo.** The two really do differ.
The vendored scheduler computes `linspace(1.0, 1/n, n)`, applies the static
shift `3σ / (1 + 2σ)` once and appends a terminal 0, which is exactly what diffusers'
`ZImagePipeline` does: it passes `get_default_z_image_sigmas` as an explicit `sigmas`
argument and `FlowMatchEulerDiscreteScheduler.set_timesteps` shifts what it was handed.
`Tongyi-MAI/Z-Image` does something else. It passes `sigmas=None`, shifts the full
1..1000 training grid in the scheduler's constructor (so its `sigma_min` is
`3 * 0.001 / (1 + 2 * 0.001)` = 0.0029940 rather than 0.001), interpolates `n + 1` points
between those already-shifted extremes and shifts a SECOND time. At 8 steps the grids
agree to a maximum |Δσ| of 5.0e-3, worst at the last step (0.3050089 against 0.3), and
the final Euler step's `dt` is −0.305009 there against −0.300000 here — a 1.7% difference
on the largest single step of the run. Small enough to be invisible by eye, large enough
to fail a parity gate at every step past the first. diffusers wins because it is the
ORACLE: Stage 3 and Stage 4 are graded against `scripts/zimage-ref-dump.py`, which drives
the diffusers pipeline and already does for the encoder, and diffusers is what
HuggingFace publishes for these weights. Moving to the official grid would mean moving
the oracle with it, and there is no reason to prefer it. Worth recording because it was
first written down backwards: the code claimed both references hand the scheduler an
explicit linspace, which is false about the official one, and candle upstream's grid —
the one this file calls wrong — was a faithful reproduction of it rather than a bug of
candle's invention. The `set_timesteps` unit test pins all nine diffusers sigmas and
asserts the last one is NOT the official repo's, so a move back is a red test rather than
a quiet drift (2026-09-07).

**An out-of-range RoPE position is refused, not clamped, and `check_size` grew a third
rule for it.** candle's Metal `index_select` kernel clamps an out-of-range id to the
table's last row — `indexing.metal`, with the comment "Force prevent out of bounds
indexing since there doesn't seem to be a good way to force crash" — while the CPU
backend errors. So a caption past 1534 tokens, or an image past 8192 px on a side, would
return a plausible picture built from the wrong rotations on the device this actually runs
on, and a CPU unit test would have caught what the shipped path would not. Two checks
close it, deliberately duplicated because they answer to different callers.
`ZImagePipeline::check_size` refuses a side past 8192 px against the shipped `AXES_LENS`
constant before a byte loads, which is what a CLI user gets and the only form available
with no config open, and it counts the token grid with checked arithmetic while it is
there. `ZImageTransformer2DModel::forward` re-checks `cap_len + f_tokens` against the
LOADED `axes_lens[0]` and the token grid against axes 1 and 2, which is the authority for
what runs and is also the only one of the two that a library caller with its own
`cap_feats` cannot skip. Neither is reachable through `xwen image` today — the encoder
truncates at 512 tokens and no admitted size exceeds the grid — and both are there
because `generate` and `forward` are `pub` and Stage 3's oracle will call them directly
(2026-09-07).

**A PNG is written to a temporary sibling and renamed.** `File::create` on the
destination truncates it before the encode runs, so a run that failed anywhere in the
encode destroyed the previous image at that path — which for a reference comparison is
the one artefact worth keeping, and the failure mode is "the run I wanted to compare
against is gone". The temp name carries the pid so two concurrent runs to one output do
not collide, and it is a sibling so the rename is within one filesystem and therefore
atomic. Pinned by a test that blocks the temporary path with a directory, the only
deterministic way to fail the encode without a broken tensor (2026-09-07).
