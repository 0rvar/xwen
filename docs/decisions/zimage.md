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
the step-0 parity gap and not for a guess before the gap exists. **Amended 2026-09-07:**
that question is answered and the answer was yes on both counts, so the weights are bf16
and everything between layers is f32 (see "The transformer's linears run on xwen's
Metal-4 tensor gemm" below); the rest of this paragraph stands. fp16 is refuted with
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

Set the same day, once the dump existed (2026-09-07, later). The dump went through
diffusers on mps rather than the official repo, `latents` being the injection point, and
it injects the CAPTION as well as the latent: `cap_feats` is the encoder's fp32 hidden
state rounded once to bf16, read by both sides, so the transformer gate grades the
transformer and Stage 2 stays the encoder's only gate; grading through two encoders would
have folded a graded gap into an ungraded one. The bars are **cosine >= 0.998 and mean
relative error <= 0.04 on the step-0 velocity**, from these figures on the 512x512 case:
the reference's own bf16 arm against its fp32 arm is 0.99956 / 0.0175, xwen is
0.99930 / 0.0205, and the two brackets, the timestep one grid point off and the caption
tokens reversed, are 0.6045 / 0.6288 and 0.8633 / 0.3259. So the bar sits at about three
times the bf16 arithmetic's own loss and a hundred times under the nearest bracket, which
separates a wrong graph from a rounding difference without flapping on the latter; the
brackets run inside the test every time, per "An agreement bar is bracketed from both
sides or it is not a bar" below. Max relative error is reported and not gated, being a
single-element statistic. Stage 4 stays reported (final latent 0.9908 against the
reference bf16 arm's 0.9947; PSNR 29.71 dB against 32.40 dB). One gate was ADDED to the
plan: the reference's final latent decoded through xwen's VAE against the reference PNG,
at **PSNR >= 60 dB**, measured 92.62. It is cheap, it is deterministic because both sides
decode in f32, and it isolates the decoder from the trajectory, which nothing else in the
plan did. The fixture is 512x512 rather than the planned 1024x1024 because the whole gate
then runs in 19 s and the dump in under two minutes, which is the fast-oracle requirement
above made concrete; the script takes `--width`/`--height` for the day a size-dependent
question arises.

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

Shipped the same evening (Arc C, `src/serve/images.rs`), with six choices made on the way
that the paragraphs above did not settle. **The images route is its own engine thread,
not a third `Job` variant.** `image-engine` sits beside the language engine with its own
bounded queue of four, its own lazy load from the cached snapshot and its own idle unload
on the same `--idle-unload` setting, so the 20 GB resident for encoder, transformer and
VAE leaves after the configured window exactly as a language model does. A third `Job`
arm would have threaded an image request through `impl Job`'s seven accessors, the
scheduler's prefill-cost closure and the KV-slot machinery, none of which has a meaning for
a render. The price is that the two engines do NOT coordinate residency: a language model
and the image pipeline can both be resident inside one idle window. That is fine for the
27B and the 35B, about 20 GB each, and it thrashes with Flash-Next, whose 111 GB is
mmap-backed and so degrades under the pressure rather than dying. Cross-engine eviction
is not taken now; the reopen condition is someone serving Flash-Next and images from one
process and measuring the stall. **The `model` rule is split by path.** `/v1/images/
generations` and `/images/generations` take the full name `Z-Image-Turbo` or nothing, the
rule every LM route follows, and a CLI alias is refused; `/proxy/openai/images/generations`
takes whatever ComfyUI's dropdown says (`gpt-image-1`, `dall-e-3`) and logs the
substitution, because that path exists for a node that cannot name the real model and
there is one image model to serve. **A missing API key on the images paths is a 403, not a
401, and a full queue is a 503 with `retry-after: 5`, not a 429**, for the reason the
amendment gives: the ComfyUI client rewrites 401, 402, 409 and 429 into comfy.org login and
credit messages before it reads the body. **`negative_prompt` and `guidance_scale` are
400s, not accept-and-drop**, reversing the "accepted and ignored" line above for those
two: Turbo is distilled to run without guidance, so a client that sent a negative prompt
would get an image that silently ignored it and no way to learn that. `quality`, `style`,
`background`, `moderation`, `user` and the rest are still dropped, since dropping them
changes nothing the client could observe. **Z-Image-Turbo stays out of `/v1/models`**,
unchanged: that list is what chat clients pick a chat model from, and the chat routes
still refuse it with the one sentence `not_servable_reason` owns. **A queued render whose
client hung up still renders.** There is no cancellation on this path; a render is
seconds to a minute and the queue holds four, so the most a hang-up can waste is a few
minutes of GPU. Not taken now; reopen if queues form in practice. A drawn seed is
`rand::random::<u64>() >> 11` so it fits in 53 bits: the seed goes back in the JSON and a
JavaScript client rounds anything wider, which would make the echoed seed not reproduce
the image it came with; a client-given seed is used as given. The route is registered
whenever the OpenAI dialect is on, and an uncached checkpoint is a 400 naming `xwen fetch
--model-size zimage-turbo`, never an in-request download (2026-09-07).

**The encoder and the transformer are both resident for the whole run.** diffusers offers
the other arrangement, `model_cpu_offload_seq = "text_encoder->transformer->vae"`, and it
exists because 8 GB plus 12.3 GB is a real problem on a 16 GB card. It is not one here:
this machine holds both, `xwen image` encodes once and then denoises eight times, and
dropping the encoder between them would buy footprint we are not short of at the price of
a reload on the next prompt. The serve route will want the opposite of offloading anyway,
since sharing one loaded Qwen3-4B between the language surfaces and the image surface is
one of the two reasons this endpoint is worth having at all (2026-09-07).

**The sigma grid is diffusers', and the official pipeline computes the same one.** The
vendored scheduler computes `linspace(1.0, 1/n, n)`, applies the static shift
`3σ / (1 + 2σ)` once and appends a terminal 0, which is exactly what diffusers'
`ZImagePipeline` does: it passes `get_default_z_image_sigmas` as an explicit `sigmas`
argument and `FlowMatchEulerDiscreteScheduler.set_timesteps` shifts what it was handed.
`Tongyi-MAI/Z-Image` gets there by a different route and lands in the same place. Its
pipeline passes `sigmas=None`, which sends its scheduler down an interpolation branch —
but the line before `retrieve_timesteps` assigns `scheduler.sigma_min = 0.0`
(`src/zimage/pipeline.py`), and with that override the branch computes
`linspace(1000, 0, n + 1)[:-1] / 1000`, which is `1 - k/n` exactly, then applies the same
single shift. At 8 steps and shift 3.0 both give
`1.0, 0.9545455, 0.9, 0.8333333, 0.75, 0.6428571, 0.5, 0.3, 0`, equal at every step.

What the override prevents is the scheduler's own CONSTRUCTOR default, and that is the
grid to recognize: left alone the constructor shifts the full 1..1000 training grid, so
its `sigma_min` is `3 * 0.001 / (1 + 2 * 0.001)` = 0.0029940 rather than 0, and
interpolating between those already-shifted extremes and shifting a SECOND time gives
`1.0, 0.9546939, …, 0.3050089` — up to 5.0e-3 off, worst at the last step, with a final
`dt` of −0.305009 against −0.300000. Nothing ships it, and candle upstream computes it
faithfully because it reproduces the scheduler without the pipeline that drives it.

diffusers is named the reference anyway because it is the ORACLE: Stage 3 and Stage 4 are
graded against `scripts/zimage-ref-dump.py`, which drives the diffusers pipeline and
already does for the encoder, and diffusers is what HuggingFace publishes for these
weights. Agreeing with the official pipeline is what makes that choice free. Worth
recording because it was written down wrong twice on the way here: first claiming both
references hand the scheduler an explicit linspace, which is false about the official one,
and then claiming the two grids differ by up to 5.0e-3, which is true of the constructor
default and false of the pipeline. The `set_timesteps` unit test pins all nine sigmas and
asserts the last one is NOT 0.3050089, so a revert to the constructor-default grid is a
red test rather than a quiet drift (2026-09-07).

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
against is gone". It is a sibling so the rename is within one filesystem and therefore
atomic. **Amended the same day:** the name is unique per WRITER and claimed by exclusive
creation, not chosen. A pid separates two `xwen image` processes and separates nothing
inside one, which the image serve route will be: two callers writing one destination would
share the name, interleave their pixels and each rename a half-written file over the
other's output. `create_new` in a bounded counter loop is what makes that impossible —
the counter proposes, the filesystem decides — and a failed rename now removes the
temporary too, since nothing will ever pick it up. Two tests: the destination survives a
write whose directory is unwritable, and a held candidate name is stepped over rather than
opened (2026-09-07).

**A path that names a diffusion snapshot is refused at the `CheckpointSource` seam, and
`inspect` gates on the FORMAT rather than on `servable()`.** `Model::ZImageTurbo`'s first
file is `model_index.json`, which is what `hub::ensure_model` returns for a
`Format::Diffusion` entry — and to `safetensors_dir` it is neither a `config.json` nor a
`.safetensors`, so it fell through to the GGUF parser. The cost was not the confusing
message: `xwen inspect --model-size zimage-turbo` resolved the entry first and downloaded
32.9 GB before failing on a magic number. Fixing it at the seam rather than at each call
site is the whole point — every checkpoint consumer already routes through
`CheckpointSource::open` (AGENTS.md, "`CheckpointSource` is the one open seam"), so one
refusal covers `generate`, `chat`, `batch`, `encode-text`, `inspect` and whatever comes
next, and `checkpoint::diffusion_snapshot_root` is the single rule for what a snapshot
path is, shared with `encode-text`'s remap onto `text_encoder/` so the two cannot drift
into the CLI accepting a shape the loader refuses. On top of that, `inspect` refuses a
diffusion ENTRY before `resolve_model`, and the predicate there is `is_diffusion()` and
not `servable()` on purpose: the Z-Image text encoder is unservable, and inspecting it is
exactly what someone reaching for `inspect` wants (2026-09-07).

**An agreement bar is bracketed from both sides or it is not a bar.** The
fused-versus-basic attention A/B shipped at 2e-3 against a real difference of 1.2e-8, and
an outside review pointed out that an arm with the `1 / sqrt(head_dim)` scale dropped
(9.1e-4) or its probabilities replaced by a uniform distribution (8.9e-5) would also have
passed. The nonzero-difference assertion that was already there catches the failure mode
AGENTS.md warns about — one kernel compared with itself — but says nothing about how
wrong the arm is allowed to be. So the rule for this repo: a tolerance is chosen by
measuring the real difference AND at least one deliberately broken variant, and the test
asserts the bar separates them. The second thing that came out of it is that the fixture
can defeat the test on its own: driven through `forward` with random weights the qk-norm
weights are uniform on ±0.1, so the logits span ±1.3e-2 and the softmax is uniform to
2e-4, and a mutation of a softmax that is already uniform is invisible. The mutation check
therefore drives the arms directly with q and k at unit RMS, which is what QK-RMSNorm
produces on trained weights (2026-09-07).

**The transformer's linears run on xwen's Metal-4 tensor gemm, with an f32 activation
stream over bf16 weights.** Every projection went through candle's steel gemm, which
measures 14-15.6 TFLOPS on this chip whatever dtype it is handed, and the linears are
about 57 of a step's 62 TFLOP, so the step ran at roughly 11.6 TFLOPS.
`crate::ops::matmul_bf16`, the cooperative-tensor kernel the language models prefill on,
measures 36.6-38.6 TFLOPS at the same shapes and the same sequence length, so the
transformer runs on it (`src/zimage/linear.rs`). The stream between layers is f32 for
four reasons, in the order they mattered. The kernel's contract is a bf16 weight against
an f32 activation returning f32, so a drop-in inside a bf16 stream pays a widen in and a
narrow out, measured at 0.10-0.12 s a step or about 9% of what the kernel wins, and the
f32 stream removes every one of them. Parity improved rather than held: step-0 velocity
cosine 0.999302 to 0.999999 and mean relative error 0.0205 to 0.0008, image PSNR 29.71 to
47.03 dB against the reference's own bf16 arm at 32.40, so xwen went from 1.6x noisier
than torch's bf16 to inside its spread. Resident memory is unchanged, the weights staying
bf16 on the device at 12.3 GB; norm weights, pad tokens and biases load f32. And the
elementwise tail costs nothing measurable in f32: the `candle` bisect arm runs the same
f32 stream and reproduces the old step time to within noise. A guard comes with it,
because the kernel stages each weight tile to f16 and a weight past f16's finite range
would be silent garbage: `ensure_weights_fit_f16` refuses any projection with |w| above
65504 at load, naming the tensor, for 0.2 s of load time, and the shipped checkpoint's
largest weight is 14.0. `XWEN_ZIMAGE_LINEAR=candle` keeps the old path as a bisect arm
that shares no matmul code with the shipped one. **Refuted: that candle's gemm was a
tuning problem.** The obvious next move was a tile config, `TILE_64_64_16_1_2` being what
its selector picks for these shapes. An A/B over the same host code and the same weights
killed it: xwen's own classic simdgroup kernel lands on candle's rate to within 5% (15.30
against 15.43 TFLOPS at T 4128) and the cooperative-tensor kernel on the same call is
2.4-2.7x faster. It is the kernel class and not the tuning, which also explains candle's
dtype-blindness, a simdgroup-matrix kernel with f32 accumulate getting little from a
narrower input type. Steps went 5.02-5.25 s to 3.06-3.59 s at 1024x1024 and 1.17-1.25 s
to 0.63-0.68 s at 512x512 ([records/zimage-perf.md](../records/zimage-perf.md),
2026-09-07).

**The VAE decodes in f32 and bf16 is refuted, with the interesting half being that it
buys nothing.** The decoder was the obvious second target after the transformer's step
time came down, and bf16 was the two-line version of it: the VarBuilder dtype in `load`
and the incoming latent's cast in `decode`. It was built, timed unprofiled, graded and
reverted. It fails the gate, the VAE-alone PSNR falling to 54.38 dB against the 60 dB bar
where f32 reads 92.62, and it is not worth the failure anyway: a 1024x1024 decode moved
from 4.93 s to 4.79 s, and 512x512 from 1.08 s to 1.09 s. Halving every byte of
convolution traffic and taking the bf16 matmul path is worth 140 ms of a 4.93 s decode,
which says plainly that the decode is not bandwidth-bound at f32. Its cost is the
structure candle's Metal `conv2d` imposes, a 9x im2col materialization plus a narrow-`n`
gemm plus an NHWC-to-NCHW permute per convolution, all of which shrink with dtype and
none of which get fewer. So the only structural fix is a direct 3x3 kernel or MPSGraph
convolution, which is a ledger item with 9.89 TFLOP and ~2.0 effective TFLOP/s behind it
(TODO.md, "Image generation"). Reopen the dtype question only on a conv path that is
bandwidth-bound, where halving the bytes would mean something; on this one it does not
([records/zimage-perf.md](../records/zimage-perf.md), 2026-09-07).
