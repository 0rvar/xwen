# Z-Image source images and masked edits

2026-09-08. Phase 1 of [the image-control PRD](../zimage-control-prd.md).

`ZImagePipeline::generate_edited` accepts RGB source pixels, a repaint mask and
strength. The VAE posterior draw is seeded independently from diffusion noise;
reference replay supplies both draws explicitly. Flux scaling happens once, in
the encoder. Mask blending uses the same diffusion noise at every step and the
next step's sigma. After decode, source pixels outside the mask are copied back.

The reference resolved one ambiguous requirement. Strength selects a fraction of
the schedule, not the nearest shifted sigma. Eight steps at strength 0.6 starts
at index 3, sigma 0.8333333, and runs five forwards. Both initialization and Euler
updates use that sigma. Strength zero is an xwen extension: source pixels pass
through, the transformer runs no forwards, and no first velocity is reported.

## Verification

The existing manual Python reference tool gained `--stage edits`. It runs the
actual `ZImageImg2ImgPipeline` and `ZImageInpaintPipeline`, replacing only their
random draws with saved tensors. The fixture uses the existing 512x512 reference
PNG, its caption, strength 0.6 and a binary centre mask aligned to latent cells.
Torch/diffusers versions, source revision and file hashes are in each generated
`meta.json`. Pixel compositing is an xwen extension and is checked separately.

| Comparison | Img2img | Inpaint |
|---|---:|---:|
| VAE encoding cosine | 1.000000000 | 1.000000000 |
| VAE encoding mean relative error | 0.0000014 | 0.0000014 |
| First velocity cosine | 0.999994749 | 0.999994749 |
| First velocity mean relative error | 0.0013993 | 0.0013993 |
| Final latent cosine, reported | 0.999938199 | 0.999995058 |
| Final latent mean relative error, reported | 0.00327289 | 0.00089108 |

`tests/zimage_edits.rs` gates encoding and the first velocity, using the existing
transformer velocity bars, and asserts exact byte preservation outside the mask.
CPU tests cover schedule boundaries, interpolation, validation and compositing.
The [parity runbook](../parity.md#image-edit-reference) gives regeneration commands.

## Not taken now

Reference images already have the requested size. Lanczos resizing may differ
slightly between image-crate and Pillow; a resize-only comparison is needed if
pixel-identical cross-runtime resizing becomes a requirement. Soft mask blur is
deliberately outside the binary-mask diffusers comparison. It runs in output
pixels before nearest-neighbour latent downsampling.

The eight-step schedule remains unchanged. A denser img2img tail is a separate
scheduler decision, reopened by a named edit that cannot be expressed at the
current strength granularity. No language-model math changed in this phase.
