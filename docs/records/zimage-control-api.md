# Image controls on the CLI and HTTP engine

2026-09-08. Phase 3 of [the image-control PRD](../zimage-control-prd.md).

The native `POST /v1/images/render` request exposes source images, masks,
strength, adapters and control maps. Images are server-local paths, data URLs
or base64. Unknown fields are errors, including fields nested inside adapters
and control settings. Image preparation and option validation precede model
loading. Omitted size follows a source image and snaps to the existing valid
grid; generation without a source keeps the 1024x1024 default.

Responses carry a `data` array with PNG base64, the seed, start step and control
map. Up to four images run sequentially, using successive seeds. The map is
returned so a client can inspect what conditioned the model. CLI flags expose
the same controls; [README](../../README.md) holds examples and fetch commands.

The OpenAI routes gained multipart edits and variations. Their transparent
mask pixels mean repaint; native requests and the CLI use white-repaint masks.
Edits default to strength 1 with a mask and 0.6 without one, and variations use
0.6. Existing generation aliases and accept-and-drop compatibility behavior
remain intact. The native endpoint is strict.

The existing image engine owns rendering and preprocessing. The bounded queue
serializes both jobs. Changed adapter or ControlNet files invalidate residency;
the outgoing pipeline is dropped before loading its replacement. Idle unload
also drops preprocessor models after a preprocess-only request. Missing cached
weights and invalid adapters return 400; inference failures return 500. The
existing image-route authentication/error convention still applies.

## Verification

Five native/multipart tests, eleven compatibility validation tests and the CLI
dependency test passed. The multipart tests parse a real SDK-shaped request,
including image bytes and alpha masks. They cover unknown fields, dependent
options, source-derived sizing and invalid seeds. Reference rendering is
covered separately by the [img2img](zimage-img2img.md),
[LoRA](zimage-lora.md) and ControlNet gates.

Two independent code reviewers and the outside Qwen reviewer examined the changes.
Review fixes preserve the published ControlNet filename through HF cache symlinks,
reject truncated official files before loading, classify merged-adapter overflow as a
request error, and attribute invalid strength, mask and blur to their own request
parameters. Seven native/multipart tests now pass, including those regression cases;
the LoRA merge tests also cover finite factors whose product overflows.

One review proposed treating final accumulated image differences as a new numerical
gate. The arc retains the existing image protocol: gate local graph outputs and VAE
decode, report the final trajectory, and test schedule/mask mechanics independently.
The outside review's strength-one concern does not change initialization: at sigma one,
the source coefficient is zero by the reference equation.

An isolated HTTP server exercised the real cached models. Native strength zero with
`n: 2` returned exact source RGB, seeds 4711/4712 and start step 8; strength 0.6 returned
start step 3. Multipart edits preserved the opaque mask region, and variations rendered.
Base, pixel-art adapter, then an empty adapter set restored the first seeded base image
exactly. Canny, pose and depth preprocessing returned PNGs; default cached ControlNet
returned its map, and a combined adapter/control request rendered successfully. Unknown
native fields and invalid preprocess types returned 400. The CLI also rendered with
source, mask, blur, adapter and control flags together. These are correctness smokes,
not server throughput measurements. The run completed 24 successful HTTP requests,
two expected 400s and two successful CLI runs; its test servers were stopped afterward.

The standing language-model gates also passed on `aee38d5`: strict, mm, three forced
decode fixtures and perplexity, for both 35B-A3B and 27B. The 35B mm cosine was
0.999618; its decode rows had only the gate's existing near-tie exceptions and zero
mismatches. The 27B had 64/64 exact decode agreements on every fixture. Absolute mean
NLL deltas were 0.001179 and 0.000243 respectively. No oracle fixtures were regenerated.

The repository harness's process-name guard rejects an idle `xwen serve` process.
To preserve the user's running server, a temporary harness copy replaced only preflight
with an empty-health check and exclusive GPU-lock ownership check, plus relocating its
imports and root path. All tier commands, environments, provenance checks and thresholds
were unchanged; the copy's diff was inspected. The server stayed running and empty.

## Not taken now

The GUI remains a separate project. No new ComfyUI, Krita or A1111 dialect is
added. Language and image model residency still use separate engines; this
phase does not add a shared memory manager. Reopen that work when concurrent
residency prevents a named workload from running.
