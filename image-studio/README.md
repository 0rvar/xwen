# Xwen Image Studio

A Tauri desktop client for xwen's image APIs. It runs separately from the inference
server and supports text-to-image, img2img, inpainting, ControlNet, preprocessing,
multiple LoRAs and parameter batches.

## Run the app

Install Bun, Rust and the Apple command-line tools. From this directory:

```sh
bun install --frozen-lockfile
bun run tauri dev
```

From the repository root, `just image-studio` builds the macOS bundle and launches it.

Pass a workspace directory at launch:

```sh
bun run tauri dev -- -- /Users/you/Pictures/study
```

Build a macOS application:

```sh
bun run tauri build --bundles app
```

The bundle is `src-tauri/target/release/bundle/macos/Xwen Image Studio.app`. Run its
executable with a workspace argument, or use `open` to start a new instance:

```sh
open -n "src-tauri/target/release/bundle/macos/Xwen Image Studio.app" --args /Users/you/Pictures/study
```

The GUI does not start the server. Run `xwen serve` separately, with the image
checkpoints already cached. See [the server README](../README.md) for model fetches
and image API configuration. A remote server works too: source images are sent as
image bytes, while LoRA paths refer to files on the server.

## Workspaces and settings

On first launch, choose a workspace folder and enter the server URL. Settings live
in `~/.config/xwen/image-studio.json`, including the server URL, optional API key,
recent workspaces and last workspace. The config file is written with owner-only
permissions. A launch argument takes priority over the saved workspace; otherwise
the app reopens the last one. The workspace menu switches between recent folders
or selects a new one.

Each launch, workspace switch or new session gets a unique directory immediately
inside the workspace. Rendered images and matching YAML files go there:

```text
study/
  20260908-143012-…/
    180326248-47-<unique>.png
    180326248-47-<unique>.yaml
    inputs/
      <sha256>.png
    batches/
      batch-<unique>.yaml
```

The YAML records the prompt, size, steps, img2img `request.strength`, requested controls and LoRAs, actual seed,
schedule start step, server URL, elapsed request time and batch coordinates. Inputs
and returned control maps are saved by content hash. Image fields in the saved
request refer to those files, so moving or deleting an original source does not
break the record. API keys are excluded. The server does not report the selected
ControlNet filename or checkpoint hashes; the YAML cannot attest to those.

Loading workspace history reads the newest 200 saved images across its sessions;
new results appear as they finish. The image viewer shows steps, strength, mask blur,
ControlNet settings and LoRA weights directly; raw metadata remains available.
Missing values in older records are labeled "Not recorded." Select an image to restore
its controls, or use it as the next source. Connection and workspace changes wait
until the current queue has stopped.

While the preview is open, use the left and right arrow keys to browse the loaded
gallery. Text fields keep their normal arrow-key behavior. The gallery grows to use
available window width, and the preview details column expands modestly on wider windows.

The image preview offers **Delete image**, which confirms before permanently
removing the PNG and YAML record. Shared input snapshots stay in place. **Delete
session** sits on each session's timestamp row in the gallery. Rows include empty
sessions and sessions beyond the gallery's 200-image preview limit.
Deleting a session confirms removal of all its images, records and input snapshots.
Deleting the current session starts a fresh one. Finish or discard queued jobs before
deleting a session; individual completed images can be deleted while rendering.

## Generation controls

Choose text-to-image, img2img or inpainting. Img2img takes a source and strength;
inpainting also takes a mask. Paint the mask in the app, or import one: white means
repaint, black means preserve. Blur is a Gaussian sigma in output pixels. The source
and mask resize to the requested output size.

Drop one PNG or JPEG onto the source or ControlNet **Choose image** field. The field
highlights while the file is over it; dropping onto an existing image replaces it.
Source imports set the output dimensions to the nearest valid size and clear the
old mask. Files are limited to 100 MB and 8192 pixels per side. Multiple files and
unsupported formats show an error without replacing the current image.

ControlNet can accompany any mode. Choose Canny, pose, depth or an already prepared
map. Preview preprocessing before rendering and use the returned map directly when
you want to keep it fixed. The server selects the Fun Union full/lite checkpoint;
the GUI controls its scale and active schedule window. The LoRA picker refreshes
the server's directory on demand and submits the returned absolute paths.

Steps default to 8. Dimensions follow the API's 16-pixel grid and token-count rule.
Seeds are limited to JavaScript's exact integer range, 0–9007199254740991. A blank
seed is resolved when planning a batch and saved explicitly. Turbo has no negative
prompt or CFG controls.

**Draft prompt with Flash-Next** turns an idea into an editable image prompt. A blank
idea asks for a new scene. It calls `Qwen3.8-Flash-Next` through the configured
server's chat API, with thinking disabled; the checkpoint must already be cached
on that server. **Use prompt** replaces the main prompt with the draft. Closing
the generator leaves the main prompt unchanged.

**Chat with image assistant** opens a conversation with the configured Flash-Next
server. The assistant receives the current output defaults and the server's live LoRA
list, and can call `queue_txt2img` to add text-to-image jobs to the same durable queue.
It cannot perform img2img, inpainting or ControlNet yet. Tool requests are validated by
the client before enqueueing; use the exact LoRA paths shown by the server.

## Batches

Add an axis for each parameter to vary. Numeric values can be comma-separated or
an inclusive `start:end:step` range. Prompts can be one per line or a JSON string
array.

| Parameter | Values | Meaning |
| --- | --- | --- |
| `seed` | `47:50:1` | Four seeds |
| `strength` | `0.4:0.8:0.1` | Five strengths |
| `steps` | `4,6,8` | Three step counts |
| `prompt` | `["a red bicycle", "a blue bicycle"]` | Two prompts |
| `loras.0.weight` | `0,0.5,1` | Three weights for the first LoRA |
| `lora` | one server LoRA path per line | Replace the active LoRA stack for each row |

Matrix mode renders every combination. Paired mode takes matching positions from
each axis; a single value repeats across rows, and other axis lengths must match.
Images per combination repeats each row with consecutive seeds. Without a seed
axis, combinations share those seeds so the parameter comparison stays matched.

The **Try with all LoRAs** button fills or replaces the `lora` axis from the
server's current LoRA list and previews one job per adapter. It replaces the
active LoRA stack so the comparison is one adapter against the same prompt and
seed; other batch axes remain in place. Each adapter uses the first active LoRA's
weight, or `0.8` when none is active. Refresh the LoRA list first after adding
files to the server.

The app validates the whole plan before sending anything and caps it at 1000 images.
The queue sends one request at a time. Stop lets the current request finish and save
before pausing; resume continues pending work. Submit can append jobs while rendering
or paused. Appending to a paused queue preserves its pause; a new submission after
the stopped queue drains starts normally. The queue allows at most 1000 unfinished
jobs, including failures until they are cleared. Failed jobs can be retried explicitly.
Each submission, including a single image, saves `batches/batch-<id>.yaml` inside
the session before rendering starts. It records the batch mode, axes, repeat count,
every resolved request and seed, shared input snapshots, job status, attempts and
output references. Each image YAML links back through its batch and job IDs.
Retry keeps the same request and seed and adds an attempt. Discard records the
pending jobs as discarded; clearing finished queue rows keeps their history.

Completed one- and two-axis batches show a **Compare batch** action above the session's
gallery. One-axis batches label each thumbnail with its tested value. Two-axis batches
use the axis with fewer values as columns and the other as rows; clicking a cell opens
the normal image preview.

Closing the app clears the in-memory queue. Its saved plan and last recorded job
states remain on disk, but automatic recovery and a parameter-exploration viewer
are not implemented. Deleting an image leaves its historical batch reference;
deleting a session removes its manifests too. The server has no render cancellation
or live step-progress API.

## Application log

Rust events and frontend console messages, handled errors, uncaught exceptions and
unhandled promise rejections go to
`~/.local/state/xwen/image-studio/logs/image-studio.log`. Server settings has a
**Show application log** button. Logs include startup, command outcomes, file-drop
diagnostics and Rust panics. A React error screen offers a reload when rendering fails.

The log rotates at 5 MiB and keeps one previous file. Its directory is owner-only.
Configured API keys and image data URLs are filtered before writing; normal events
omit request bodies and prompts. Frontend forwarding is bounded and buffered, so an
abrupt process kill can lose pending messages. Log forwarding failure reports to the
original console without recursively logging itself.

## Development checks

```sh
bun run check
bun test src/domain.test.ts
bun run test:e2e
cargo test --manifest-path src-tauri/Cargo.toml
```

Playwright needs Chromium installed (`bun playwright install chromium`). Its
tests run the explicit preview bridge with fixture responses. `bun run dev:preview`
opens that same UI for browser-only development; it does not generate images or
write workspace files. Normal desktop builds always use the Rust bridge.

Architecture, verification and remaining limits are in
[the implementation record](../docs/records/image-studio.md).
