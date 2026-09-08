# Z-Image transformer adapters

2026-09-08. Phase 2 of [the image-control PRD](../zimage-control-prd.md).

Adapters merge into a fresh copy of the transformer's weight planes at load.
Each pair contributes `weight * alpha/rank * B@A`; an absent alpha means rank,
so its multiplier is one. Base weights and the accumulated deltas stay f32 until
the final requested dtype conversion. The decode loop has no adapter operations.

The shipped transformer already stores separate Q/K/V planes. Split adapters
apply directly; fused QKV adapters are split by output rows. Missing components,
duplicate or overlapping targets, nonfinite factors, wrong shapes and targets
the transformer never consumes are errors. The merged weights still have to fit
the existing f16 staging range. Refiner and embedder projections are valid
targets and are shared with ControlNet.

The image engine keys residency by the ordered adapter set, weights and file
identity. A changed set drops the resident pipeline before loading the base
again. An empty set restores the unadapted model. Paths are canonicalized, and
file size, modification time, inode and change time detect ordinary replacement
at the same path.

## Verification

The real fixture is `tarn59/pixel_art_style_lora_z_image_turbo`, revision
`0a5092d1619664d94a5a36784f92db84b3ae62bd`, file
`pixel_art_style_z_image_turbo.safetensors`. It is a 170 MB rank-32 split-QKV
adapter with 480 tensors and no alpha entries. The reference uses diffusers'
load, set and fuse APIs at weight 0.8, over the existing 512x512 caption/noise
fixture. Adapter bytes remain in the HF cache; the reference manifest records
the complete SHA256.

`tests/zimage_lora.rs` found the first Q projection's merged plane and its delta
exactly equal to the reference. First velocity cosine was 0.999545422 and mean
relative error 0.0158865, inside the existing transformer gate. Final latent
cosine was 0.994706832, mean relative error 0.0519386, and image PSNR 32.527 dB;
those accumulated differences are reported, as in the original image gate.
CPU tests cover alpha scaling, partial attention updates, fused remapping,
unknown targets and malformed adapters.

## Not taken now

Runtime adapter switching, DoRA and other adapter families are not implemented.
Underscore-encoded trainer keys must be exported to the supported diffusers or
Comfy-style dotted names. Reopen a format when a named adapter requires it.
The file identity is an ordinary filesystem change detector, not a content
addressed adapter registry.
