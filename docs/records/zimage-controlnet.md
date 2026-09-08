# Z-Image Fun Union control

2026-09-08. Phase 4 of [the image-control PRD](../zimage-control-prd.md).

The side model supports the official full and lite 2601/2602 8-step files.
Lite is preferred when both are cached. Plain, non-distilled files have the
same tensor shapes and require CFG, so the loader checks the official filename
and byte length as well as tensor structure. An override path retains its
official basename. `XWEN_CONTROLNET_FILE` chooses a file explicitly.

The core uses the base transformer's actual embedders and refiners, so adapters
on those modules affect control too. Main control residuals follow blocks
0,2,...,28 for full and 0,10,20 for lite. Both control-refiner residuals reach
the generator. The 132-wide control embedder uses candle's projection path;
its input width does not meet the tensor GEMM's 32-element requirement. The
large block projections use the existing tensor path.

## The author is the control oracle

Source inspection found three different graphs. Diffusers runs joint image and
caption control attention but does not return its refiner residuals to the
generator. ComfyUI applies refiner residuals but keeps the control main stack
image-only. The checkpoint author's VideoX-Fun implementation runs joint control
attention and applies both refiner residuals. This port follows the author
throughout rather than combining pieces of the two other implementations.

The control context has 33 channels: 16 encoded control, one keep-mask and 16
encoded masked source. Both VAE encodes use the posterior mode. Missing source
conditioning is zero latent. A source is masked in normalized pixel space
before encoding, so repainted pixels are grey, not zero latent. Trained
inpainting uses that conditioning without tier-one per-step latent restoration;
the final source-pixel composite remains an explicit xwen extension.

Control runs only inside the requested half-open window of the full schedule.
The default is `[0,0.8)` at scale 0.75. Scale zero skips the side graph exactly.
No CFG or negative prompt path is added.

## Verification

The manual reference tool imports the author's unmodified modules from a
SHA-pinned external checkout and records source checksums in its fixture
manifest. The first real run used lite 2602 at scale 0.75, full window, eight
steps and the existing 512x512 caption/noise fixture.

| Comparison | Plain control | Trained inpaint |
|---|---:|---:|
| Prepared context cosine | 0.999999999 | 0.999999999 |
| Prepared context mean relative error | 0.0000022 | 0.0000025 |
| First velocity cosine | 0.999992841 | 0.999998861 |
| First velocity mean relative error | 0.0024866 | 0.0010703 |
| Final latent cosine, reported | 0.982228703 | 0.999980940 |
| Final latent mean relative error, reported | 0.1211703 | 0.0023558 |

Image PSNR was 30.252 dB for plain control and 48.830 dB inside the trained
inpaint region. The reference latent through xwen's VAE scored 92.174 dB in
both cases. Prepared contexts and first velocities pass their gates; accumulated
image differences are reported. Zero-scale velocity is bitwise equal to the
base path. Tiny-model tests independently prove the before/after projection
recurrence and that refiner-only residuals reach the final generator output.

The full 2602 file passed the same author comparison, using all fifteen main
control blocks and two refiners:

| Comparison | Plain control | Trained inpaint |
|---|---:|---:|
| First velocity cosine | 0.999998805 | 0.999994986 |
| First velocity mean relative error | 0.0011841 | 0.0015374 |
| Final latent cosine, reported | 0.999994628 | 0.999987307 |
| Final latent mean relative error, reported | 0.0024032 | 0.0024049 |
| Image PSNR, inpaint region only for inpaint | 57.933 dB | 51.007 dB |
| Reference-latent VAE PSNR | 94.077 dB | 93.108 dB |

Its prepared-context results match lite, and its zero-scale velocity is also
bitwise equal to base. These runs use author revision
`968f0e2192ba4c7a12868bf36d73260d135424ca` and checkpoint revision
`5155fc56d17821007d6f62ac192c09e0f0e72016`. The loader accepts the 2601 filenames
with the same architecture, but only 2602 was numerically compared.

## A bounded control-map study

One 512x512 case, eight steps, scale 0.75, window `[0,0.8)`. Canny and depth maps
extracted from the existing fisherman fixture changed a standing composition into
a seated one. Canny also introduced conspicuous limb and clothing artifacts. These
outputs establish that depth is consumed; they do not establish a general quality
ranking between preprocessing modes.

A hand-drawn OpenPose skeleton specified a centered person with horizontal arms and
visible feet. At seed 47, the prompt "A full-body photograph of a man, both feet
visible, plain studio background" produced cropped standing legs without control.
Both full and lite produced a coherent full-body arms-out image with the synthetic
map. Full also followed the broad seated geometry of the DWPose map extracted from
the fisherman. The generated pose is therefore not explained by the prompt alone.

Prompt sensitivity matters. Earlier lite runs with "standing naturally" left the arms
down; changing the prompt as well as the checkpoint initially made full appear to be
the only working variant. Repeating lite with the exact neutral prompt resolved that
confound. Keep lite as the cached default. These few images justify supporting
hand-drawn maps and do not justify opening CFG or claiming reliable pose adherence
across prompts and seeds.

The [study manifest](../../tests/fixtures/zimage-control-study/manifest.json) preserves
the exact settings, checkpoint revisions and image hashes. Its images include the
[uncontrolled baseline](../../tests/fixtures/zimage-control-study/neutral-baseline.png),
[synthetic map](../../tests/fixtures/zimage-control-study/synthetic-pose.png),
[lite result](../../tests/fixtures/zimage-control-study/lite-neutral-synthetic.png),
[full result](../../tests/fixtures/zimage-control-study/full-neutral-synthetic.png), and
[extracted-pose result](../../tests/fixtures/zimage-control-study/full-neutral-pose.png).
They are manual-study evidence, not pixel goldens for automated tests.

## Cost protocol

The current figures live in [perf-state.md](../perf-state.md). The pinned binary was
built in a detached worktree at `aee38d5`; its SHA256, per-run arguments, power lines,
intervals and aggregation are in the [timing artifact](../../tests/fixtures/zimage-control-study/timing.json).
Base, lite 2602 and full 2602 ran serially at 512x512 with the existing caption/noise
fixture, eight steps, scale 0.75 and window `[0,0.8)`. The supplied
[control map](../../tests/fixtures/zimage-control-study/timing-control.png) bypasses
preprocessing. Use `--cap-feats` and `--latents` from
`tests/fixtures/zimage-transformer/512x512-p1-s0`, plus `--control-type none` for control
arms, and select the side file with `XWEN_CONTROLNET_FILE`.

All `XWEN_ZIMAGE_*` overrides were cleared. One complete base/lite/full round was
recorded as warm-up after the first base process paid a cold shader compilation cost.
Three further complete rounds supply the medians; each round and the final baseline
anchor follow 60 seconds idle. The GPU lock was exclusive, peer builds stopped, the
existing server was empty and the smoke server was stopped. Fixed base/lite/full order
means full follows two short model runs; its later-step spread is retained, not
filtered. The closing warm anchor stayed below the protocol's drift threshold.

The first step's readback also drains the pending control VAE encode, so its timer is
not an isolated transformer measurement. Logged intervals round to 0.01 seconds, and
CLI total/load times to 0.1 seconds. These runs price short 512x512 generation with
control active for seven of eight steps. They do not price sustained 1024x1024, trained
inpaint, a full control window, text encoding or preprocessor latency.

## Not taken now

The non-distilled ControlNet and CFG stay conditional. Reopen them for a named
pose or inpaint composition whose 8-step artifacts are the blocker. A broad pose
quality study remains unmeasured; reopen it for a client workflow that needs a
reliability estimate beyond the bounded case above. Tile upscaling and base
Z-Image remain outside this PRD.
