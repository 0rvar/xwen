# Image control on Z-Image-Turbo: img2img, inpainting, LoRA, ControlNet — PRD

Working doc, opened 2026-09-08. This is the product spec for taking `xwen image` and the
images route from text-to-image only to the full set of controls the model family
supports. TODO.md's "Image generation" section is the ledger pointer once the items are
promoted; this doc is the substance. When an arc ships, its decisions migrate to
[decisions/zimage.md](decisions/zimage.md) and the story to [log.md](log.md) per the doc
system. The architecture reference stays [zimage.md](zimage.md).

Why now: a 125-image prompt study on 2026-09-08 (`~/Pictures/batch/`, guide in
`PROMPT-GUIDE.md` there) found the limits of prompting on this model. Setting, lighting,
pose phrasing and local markings respond to text; framing does not ("full body" cropped
the feet in ~90 of 96 attempts), identity drifts between prompts, seeds barely change the
image, and pose beyond "bent over, looking back" is not enforceable. Every one of those is
what the controls below exist for.

## Where we are

The list below is the starting point on 2026-09-08. Phases 1–5 now have implementations
and reference gates: [edits](records/zimage-img2img.md), [LoRA](records/zimage-lora.md),
[CLI and HTTP](records/zimage-control-api.md), [ControlNet](records/zimage-controlnet.md)
and [preprocessors](records/zimage-preprocessors.md). Their records own the measured
results and limitations. Phase 6 remains conditional; the GUI remains outside this PRD.

- `xwen image` renders text-to-image on Metal: Qwen3-4B encoder, 8 flow-matching steps
  from seeded noise, VAE decode. `--latents` replaces the step-0 noise with a tensor from
  a file, `--cap-feats` replaces the encoder output. Both are parity instruments, not
  user features.
- `POST /v1/images/generations` serves the same in the OpenAI shape, with `steps`,
  `num_inference_steps`, `seed`, `rng_seed` as extension fields; `negative_prompt` and
  `guidance_scale` are 400s because Turbo has no guidance
  ([decisions/zimage.md](decisions/zimage.md), Arc C).
- The VAE encoder is built (`src/zimage/vae.rs`, `AutoEncoderKL::encode`) and nothing in
  the pipeline calls it. img2img and inpainting need it and no new weights.
- `controlnet_block_samples` in the transformer is an unused hook ([zimage.md](zimage.md)).
- The decision that a ComfyUI node pack of our own is built "only when the user names a
  composition that needs it: img2img, inpainting, LoRA or an upscale chain"
  ([decisions/zimage.md](decisions/zimage.md), 2026-09-07). This doc names three of the
  four but does not take the node pack: the client for these controls is a GUI of our
  own, planned separately (see "Clients"), so the condition is superseded rather than
  met. Reopen if driving xwen from inside ComfyUI or Krita becomes something named.

## Corrections to the naive plan

The first sketch was "Turbo for generation, ControlNet for img2img, inpaint and control".
Three things are wrong with it and they shape the phases:

1. img2img does not need the ControlNet. It is plain Turbo with a VAE encode and a start
   partway into the schedule.
2. Inpainting has two tiers. Masked-latent blending on plain Turbo needs no new weights.
   The ControlNet's inpaint mode is the trained one and is better at large holes. Ship the
   cheap tier first.
3. Pose and depth extraction are separate models, not part of the ControlNet. Canny needs
   no model. The union ControlNet does not care where the map came from.

And one consequence that is easy to miss: the non-distilled ControlNet needs CFG and 20
to 40 steps. xwen has no CFG path since Turbo bakes it in. The 8-step re-distilled
ControlNet variant avoids that entirely, so it ships first; CFG is a later phase with a
reopen condition, not a prerequisite.

## The model facts that bind the design

Read off the checkpoints, the diffusers and ComfyUI sources and the maintainers' own
threads on 2026-09-08. Sources at the end.

- Turbo and base Z-Image share one architecture and one weight shape; Turbo is base plus
  Decoupled-DMD distillation and RL post-training. That is what bakes in the guidance and
  costs the seed diversity. Base runs 28-50 steps at guidance 3-5 with working negatives.
- `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1` is one checkpoint for canny,
  depth, pose, MLSD, HED, scribble, grayscale and inpaint. No mode selector; feed the map.
  Full: a copy of 15 of the 30 transformer blocks plus the 2 refiners, 3.36B params, 6.71
  GB bf16, injected as residuals into every second main block. Lite: 3 blocks, 2.02 GB,
  every tenth block, weaker and "more natural" by the maintainer's own description. Tile
  for upscaling is a separate file. The `-8steps` files are the same size, re-distilled
  on top of 2.1.
- The control image is VAE-encoded once into latent space, not run through a pixel-space
  hint encoder. **Corrected during implementation, 2026-09-08:** inpaint mode takes
  33 latent channels: 16 control, one keep-mask, and 16 masked source. The source is
  grey-masked in pixel space before VAE encoding; both encodes use the posterior mode.
- The ControlNet runs the base transformer's embedders and refiners by reference
  (`from_transformer` shares `t_embedder`, `all_x_embedder`, `cap_embedder`,
  `rope_embedder`, `noise_refiner`, `context_refiner`). A LoRA on any of those changes
  ControlNet behaviour. It is trained against Turbo; base has its own checkpoint.
- It runs on every step. Neither diffusers nor ComfyUI expose a start/end step window
  for this model; fal's hosted endpoint does (`control_start`, `control_end` default 0.8).
  With no control image the pipeline has nothing to encode, and the inpaint-capable
  files substitute flat grey. It cannot be an always-on default.
- Plain 2.1 "lost acceleration capability during training, requiring more steps and cfg"
  (README); the maintainer's grid was 9-40 steps at scale 0.65-1.00. The 8-step variant
  is reported good on canny and artifact-prone on the other modes (HF discussion #14).
- Synthetic pose maps (3D pose editor, hand-placed OpenPose) are reported as not
  recognized at all, while extracted ones work (HF discussion #28, open, unanswered).
- LoRA gotcha: adapter ecosystems use both split and fused QKV. The shipped xwen
  transformer uses split `to_q`/`to_k`/`to_v` planes; fused adapter deltas need a row
  remap. Unknown or unapplied deltas must fail instead of silently doing nothing.
- Size rule stays: both dimensions a multiple of 16, `(w/16)*(h/16)` a multiple of 32.

## End state

One image pipeline, Z-Image-Turbo the only generator. Every request is text-to-image at
its core with optional inputs that change how the latent starts and what conditions the
transformer. Which inputs are present decides the mode; there are no separate mode
switches.

| Inputs present | Mode | Weights beyond Turbo |
|---|---|---|
| prompt | generation | none |
| + `loras` | generation with adapters | the adapters |
| + `init_image`, `strength` | img2img | none (VAE encoder already built) |
| + `init_image`, `mask` | inpainting, tier one | none |
| + `control` | ControlNet | Fun Union, 8-step |
| + `init_image`, `mask`, `control` | inpainting, tier two | Fun Union, inpaint mode |

### Generation

Unchanged. Guidance stays baked in. `negative_prompt` and `guidance_scale` stay 400s
until the CFG phase, if it happens.

### LoRA

One or more adapters on the transformer, a weight each. Merge into the weight planes at
load (one matmul path, no runtime cost, but a reload to change the set) rather than a
runtime low-rank side path; revisit if adapter switching per request turns out to
matter. The QKV remap is part of the loader, with a test that loads a split-QKV adapter
and asserts the attention planes changed. Adapters touching embedders or refiners are
allowed and documented as also changing ControlNet output.

### img2img

VAE-encode the init image with the Flux `shift_factor`/`scaling_factor` on the way in.
**Corrected against the executable reference, 2026-09-08:** strength selects
`floor(N - N * strength)` as the start index. Initialize using that step's shifted
sigma, `x = (1 - sigma) * z_init + sigma * noise`. Eight steps at strength 0.6
starts at index 3, sigma 0.8333333, and runs five steps. The API accepts any float
in `[0,1]` and reports the actual start index; zero is source pass-through. The init
image is resized to the request size, or the
request size defaults to the image's, snapped to the size rule. This retires the
user-facing role of `--latents`, which stays as the parity instrument it is.

### Inpainting, tier one

img2img plus a mask. After every step, outside the mask the latent is overwritten with
the source latent noised to that step's level; inside it is the model's output. Mask is
downsampled to latent resolution, white means repaint, with an optional blur radius in
pixels. After decode, original pixels are pasted back outside the mask so the VAE
round-trip does not soften them. `strength` defaults to 1.0 here.

### ControlNet

The Fun Union checkpoint as an optional side model, loaded per request and unloaded with
the pipeline on idle. Request carries the map, a `scale` (default 0.75, usable range
0.65-1.0), and a start/end window in fractions of the schedule, default 0.0-0.8, which
diffusers lacks and fal has; ours is a loop-level choice, free to add. The map is
VAE-encoded once and cached for the run. Control blocks run at each step inside the
window and add residuals into the main blocks at the configured indices. Inpaint mode is
selected by the presence of `init_image` and `mask` alongside `control`, builds the
33-channel input, and is tier two above.

The 8-step variant is the only one loaded until the CFG phase. Full and lite are both
supported, chosen by which file is present, lite recommended first.

### Preprocessors

Turn a photo into a control map. Canny is a few lines and ships with the ControlNet.
Pose (DWPose or OpenPose) and depth (Depth Anything) each need a small model and their
own port; they come after. Until then the API accepts pre-made maps, which is how the
diffusers pipeline works too. Preprocessing is its own endpoint as well as a field on
the generate request, so a client can inspect or hand-edit a skeleton before rendering.

### API

Two surfaces on the existing image engine thread. The native one is primary and is
designed for a single client we control; the OpenAI one stays for SDKs and scripts.

The OpenAI images endpoints for compatibility, as today plus `edits` (image + mask +
prompt, which is tier-one inpainting at strength 1.0, and img2img when the mask is
absent) and `variations` (img2img at a fixed strength). They cannot express LoRAs,
strength, control or the window, so those take defaults. No further work to court
third-party apps: no field aliases beyond the ones already accepted, no extra path
prefixes, no A1111 dialect. That was surveyed on 2026-09-08 and dropped in favour of
our own client.

A native endpoint, `POST /v1/images/render`, one JSON shape, images as base64 or as a
path the server can read:

```json
{
  "prompt": "...",
  "width": 896, "height": 1152, "steps": 8, "seed": 4711,
  "loras": [{"name": "...", "weight": 0.8}],
  "init_image": "...", "strength": 0.6,
  "mask": "...", "mask_blur": 16,
  "control": {"image": "...", "preprocess": "none|canny|pose|depth",
              "scale": 0.75, "start": 0.0, "end": 0.8}
}
```

Response: the PNG as base64, the seed, the step the schedule started from, and for
control requests the preprocessed map so the client sees what the model saw. Errors in
the OpenAI envelope like the rest of the route. Unknown fields are 400s, not ignored;
that is the opposite of the compatibility route and deliberate, since this shape is ours.

`POST /v1/images/preprocess` takes `{"image": "...", "type": "canny|pose|depth"}` and
returns the map.

The CLI grows the same fields as flags: `--init`, `--strength`, `--mask`, `--mask-blur`,
`--lora name[:weight]` repeatable, `--control path`, `--control-type`, `--control-scale`,
`--control-window`.

### Clients

The native endpoint's client is a GUI of our own, xwen-gui, planned as its own doc and
not part of this PRD. What this PRD owes it is the contract above and one response
shape that carries everything the GUI will show: the image, the seed, the start step,
and the preprocessed control map. The requirements the prompt study surfaced for that
GUI are recorded here so the endpoint does not paint it into a corner: a grid keyed on
one prompt across seeds and on one seed across prompt variants, since that is how this
model is judged; a mask painter for the inpaint path; a control-map preview for pose.
Nothing in the endpoint should assume one image per request or hide the map.

The OpenAI route's clients are the openai SDKs with `base_url`, LiteLLM, and the MCP
image servers that take a base URL, so Claude Code can render through it. Those already
work with the route as it is.

## Phases

Each phase is an arc with its own record. Order is by what it unlocks per unit of work.

1. **VAE encode in the loop, img2img, tier-one inpainting.** No new weights. Parity:
   diffusers `ZImageImg2ImgPipeline` and `ZImageInpaintPipeline` at matched seed and
   strength, graded the way the pipeline arc graded text-to-image
   ([parity.md](parity.md)). Unlocks composition from a blob painting and local edits.
2. **LoRA loader with the QKV remap.** Parity against diffusers with the same adapter
   and weight. Unlocks identity and style adapters; the ControlNet interaction is
   documented, not tested, until phase 4.
3. **Native endpoint and CLI flags for phases 1-2**, plus `edits` and `variations` on
   the OpenAI route.
4. **ControlNet, 8-step union, canny preprocessor.** **Oracle corrected during
   implementation:** the checkpoint author's VideoX-Fun graph is authoritative.
   Diffusers omits generator refiner injections and ComfyUI uses image-only control
   attention; the author uses both refiner injections and joint caption/image
   attention. Gate against the author at scale 0.75, full window.
   Then the window on top. Then inpaint mode, tier two.
5. **Pose and depth preprocessors**, and the preprocess endpoint.
6. **CFG and the non-distilled ControlNet.** Only if phase 4's 8-step variant falls
   short on pose or inpaint in practice, which the discussion threads say it may. Reopen
   condition: a named composition where the 8-step file's artifacts are the blocker.

The GUI is not a phase here. It starts once phase 3 gives it an endpoint to talk to and
runs as its own arc with its own doc.

The 2026-09-08 client arc is [Image Studio](records/image-studio.md), a Tauri app
under `image-studio/` with workspaces, image controls and parameter batches.

## Risks and open questions

- Synthetic pose maps may not work (discussion #28). Test a hand-drawn OpenPose map in
  phase 4 before building the pose preprocessor around the stickman use case. If it
  fails, the workflow is a posed reference photo through DWPose.
- The 8-step ControlNet may be canny-only in practice. Phase 4's record answers this
  with the same prompt study protocol as the batch, and phase 6's reopen condition is
  written against it.
- Memory. Turbo is ~20 GB resident on the engine thread; the full ControlNet adds 6.7
  GB, lite 2 GB, LoRAs merge at zero cost. Idle unload covers it; a request that needs
  the ControlNet while a language model is resident is the same fight the images route
  already has and is not solved here.
- Per-step cost with the full ControlNet is roughly +50% by parameter count; nobody
  publishes a measurement. Phase 4 measures it per [benching.md](benching.md) and
  [perf-state.md](perf-state.md) gets a row.
- Whether img2img at eighth-step granularity is fine enough. If not, the fix is a
  denser schedule for the img2img tail, which is a scheduler change and not free.

2026-09-08 verification answers: the [control record](records/zimage-controlnet.md)
contains a saved one-seed study where both lite and full follow a synthetic OpenPose
map, full follows an extracted seated pose, and depth changes composition. Prompt
sensitivity and Canny anatomy artifacts remain visible, so this is not a general
quality guarantee. The same record holds the pinned cost protocol, with the current
figures in [perf-state.md](perf-state.md). No tested case requires reopening CFG.
Finer img2img schedule control and shared language/image residency remain conditional
on an actual client workload.

## Not in scope

- Z-Image base as a second generator. Same shapes, but a different resident set, CFG,
  and its own ControlNet and LoRA ecosystem. Reopen when Turbo's seed diversity or
  fine-tunability is the blocker for something named.
- Z-Image-Edit and Omni-Base. Both "to be released" on the official README as of
  2026-09-08; there is no checkpoint. Instruction editing waits for one.
- Tile upscaling. Separate checkpoint, separate arc, and the prompt study did not need it.
- Training or fine-tuning of any kind.
- The GUI itself, and any ComfyUI or Krita integration. See "Clients".

## Sources

- [Tongyi-MAI/Z-Image README](https://github.com/Tongyi-MAI/Z-Image): model zoo, release
  status, Turbo vs base settings.
- [Decoupled DMD paper](https://arxiv.org/abs/2511.22677) and the
  [Z-Image paper](https://arxiv.org/abs/2511.22699).
- [alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1](https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1):
  README update log, training details, scale guidance; discussions #10, #14, #18, #22,
  #23, #28 for the failure reports quoted above.
- diffusers `pipelines/z_image/` and `models/controlnets/controlnet_z_image.py`: pipeline
  signatures, the shared-module coupling, the absence of a control window, the 33-channel
  inpaint layout.
- ComfyUI `comfy_extras/nodes_model_patch.py`: `ZImageFunControlnet` inputs, the
  every-step injection schedule `div = round(30 / cnet_blocks)`, the grey substitute.
- [fal.ai Z-Image Turbo endpoints](https://fal.ai/models/fal-ai/z-image/turbo/controlnet):
  `preprocess`, `control_scale`, `control_start`, `control_end`, `strength`,
  `mask_image_url` as the reference request shape.
- `capitan01R/Comfyui-ZiT-Lora-loader`: the split-vs-fused QKV remap.
