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

## Not taken now

The non-distilled ControlNet and CFG stay conditional. Reopen them for a named
pose or inpaint composition whose 8-step artifacts are the blocker. A synthetic
pose-map quality study remains a separate product judgement; numeric graph
parity alone does not establish how well the trained model follows a hand-drawn
skeleton. Tile upscaling and base Z-Image remain outside this PRD.
