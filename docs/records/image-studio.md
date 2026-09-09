# Image Studio uses the image APIs from a Tauri desktop app

2026-09-08. The user requested a desktop GUI in a repository subdirectory, with
home-directory configuration, selectable workspaces, CLI workspace selection,
per-session image output and generation metadata, all image controls, and range or
matrix batches. This is the separate client arc anticipated by the
[image-control PRD](../zimage-control-prd.md#clients).

## Application boundary

`image-studio/` is an independent Bun/Vite/React frontend and Tauri v2 Rust package.
Its Cargo lockfile and build products are separate from the inference engine. It
does not link xwen, load model weights, alter kernels or start a server. Rust owns
HTTP and disk writes. The webview invokes typed commands, which lets the same app
use localhost or a remote xwen without requiring browser CORS configuration.

Tauri plugins provide native dialogs and Rust file logging. Dialogs are granted
`dialog:allow-open`; log ingestion uses an application command rather than plugin
permissions in the webview.
Images travel as data URLs through the bridge; there is no broad home-directory
asset-protocol scope. One reqwest client handles the server connection, with
redirects disabled and separate connection and render timeouts. Config lives at
`~/.config/xwen/image-studio.json`, written atomically with mode 0600. Native CLI
arguments override the saved workspace. A missing last workspace leaves the app
able to choose another; corrupt configuration is reported rather than overwritten.

The UI separates source editing from optional ControlNet conditioning, so all
native API combinations remain available. Mask painting produces the API's white
repaint/black preserve mask. LoRA listings refresh on request and the GUI uses
`path` for selection. Control preprocessing can be previewed and the resulting map
submitted with `preprocess: none`. The server's chosen full/lite checkpoint is not
a GUI selection because the API does not expose it.

## Output and provenance

Every session has a unique directory under its workspace. Every completed image has
a neighboring YAML sidecar containing the request and returned seed/start step,
output dimensions/hash, timestamps, elapsed request time and batch coordinates.
Source, mask and control inputs are copied under `inputs/`, deduplicated by SHA-256;
the saved request refers to those relative paths. Returned control maps are saved
too. Metadata excludes authentication and does not invent server weight versions.

YAML is authoritative. PNG supports textual metadata, including UTF-8 iTXt, but
embedding would modify the server's image and some editors discard ancillary data.
The sidecar is readable, handles nested controls and adapters, and records references
to the input snapshots. Embedding a compact summary remains an option if a sharing
workflow needs a single file. [PNG text-chunk specification](https://www.w3.org/TR/png-3/)

The gallery reads up to 200 of the newest app-generated image/sidecar pairs across
workspace sessions. Reusing settings restores the saved scalar controls and input
snapshots; using an output as source begins a new edit. Old sessions remain on disk
when workspaces switch.

## Batch semantics

The planner is a pure TypeScript module. An axis is a parameter name and either a
numeric list/range or prompt list. Matrix mode forms a Cartesian product; paired
mode broadcasts singleton axes and otherwise requires equal lengths. Each
combination can repeat with consecutive seeds. The same seed sequence is reused
across parameter combinations unless seed itself is an axis. Empty seeds resolve
once per plan, then every request carries its seed explicitly.

The complete plan is validated before the first HTTP request. Invalid dimensions,
out-of-range controls, nonexistent LoRA indices, duplicate axes, unsafe seeds and
plans above 1000 images are refused. Each queue item snapshots its request, and
requests run sequentially. A stop takes effect after the active request saves its
result; errors remain visible and can be retried. No server cancellation or
denoising percentage is claimed, because neither exists on the API.

## Verification

The TypeScript check and 15 domain tests pass. They cover request modes, source
dimension snapping, range parsing, Cartesian products, paired-axis broadcasting,
matched repeat seeds, immutable controls, invalid-cell rejection and metadata
restoration. Nine Rust tests pass, covering config permissions and corrupt-file
refusal, CLI precedence, unavailable workspaces, authenticated HTTP and errors,
busy-state recovery, atomic publication, thumbnails and provenance checks. Gallery
asset hashes are cached only within one scan; subsequent scans detect modified
inputs. Checking a draft server connection leaves the saved settings unchanged.

Five Playwright flows pass through the explicit development bridge: first-launch
setup and picker cancellation; testing a connection without saving; mask editing,
ControlNet and LoRA request construction; matrix execution with pause/resume; and
paired validation, retry, restored assets and workspace switching. Mask tests check
the canvas fits its stage with the right aspect ratio, then inspect the exported
white stroke and black corner. The combined render test inspects the submitted
mask, fixed control map and absolute server LoRA path. Production assets contain
none of the development bridge's fixture markers. The initial, batch and mask
screenshots under `/tmp/xwen-image-studio-*.png` were visually inspected.

An opt-in real-server test rendered a 512-square image at seed 47 through the same
Rust service the GUI calls, parsed the saved YAML and checked the output hash, then
rendered a zero-strength img2img request from that output and compared every RGB
pixel. It passed. Its copied session, PNGs, YAML and input snapshot are under
`/tmp/xwen-studio-smoke/results`; the owned server on 5242 was stopped afterward.
The user's server on 5241 was not restarted or modified.

The release macOS app bundle builds. A native process created an Image Studio
window, but this execution environment denied macOS window capture and logged
WebKit service/website-data permission errors. Native interaction is therefore not
claimed as verified. The test processes were stopped and no real user config was
written by that launch. Browser automation checks the webview UI separately.

Two code reviews covered the backend/planner and frontend independently of their
implementers, followed by the external Qwen review. Fixed findings included stale
image/control previews, source-grid snapping, a cropped mask canvas, unnecessary
full-resolution gallery payloads, a backend-only size restriction, repeated input
hashing and connection checks that saved settings. The Qwen report's remaining
claims were checked against the code and real render test: control maps are decoded
and validated, saved assets must exist, and the supposed inverted start-step check
was correct. No actionable findings remain. The final bundle was rebuilt after
the fixes; docs-check and the unchanged ledger pass.

## Gallery cleanup and prompt drafts

2026-09-08. The follow-up adds confirmed permanent image and session deletion,
submission into an active queue, a fix for submission after stop, and a Flash-Next
prompt generator. Session management scans every recognized session rather than
deriving its list or counts from the gallery's newest 200 images. Removing the
active session rotates to a new one. Image deletion keeps shared input snapshots;
session deletion removes the whole folder. Files outside the selected workspace,
symlinked session/output paths and foreign records are refused. Finder metadata is
allowed, and an owned orphan YAML record does not prevent session cleanup.

The queue preserves request snapshots and FIFO ordering when more jobs arrive.
Appending while paused preserves that pause; a new submission after the stopped
queue drains starts without a separate resume. Unfinished work, including failed
jobs, is capped at 1000. Workspace/server changes and session deletion remain
blocked while running or pending work exists.

The prompt dialog calls the configured server's `/v1/chat/completions` endpoint
with the exact `Qwen3.8-Flash-Next` model, `stream: false`, a 512-token cap and
`chat_template_kwargs.enable_thinking: false`. It reads only message content and
rejects empty or truncated answers. The draft is editable and replaces the main
prompt only on explicit use. Late responses after closing the dialog are ignored.
The server must already have Flash-Next cached; its error is shown without changing
the current prompt or falling back to another model.

TypeScript and the 15 planner tests pass. Eleven browser flows cover the original
controls plus FIFO appends, paused appends, restart after drain, canceled/confirmed
deletion, active-session rotation and editable prompt drafts with error handling.
The Rust suite checks actual temporary-file deletion, workspace ownership, traversal
and symlink refusals, 205-image session counts, retained shared inputs, rendering
concurrency and the authenticated Flash-Next request/response contract. Browser
fixtures do not claim model quality or native window interaction.

All 12 Rust tests pass (the opt-in image-render smoke test remains ignored). A live
chat request to the existing server returned a usable telescope/observatory prompt
from `Qwen3.8-Flash-Next`, with a normal stop and zero reasoning tokens. The response
is saved at `/tmp/xwen-wishlist-live-prompt.json`. The server was not restarted.
Frontend and backend reviews plus the external Qwen review ran; fixes included
late prompt responses, stale history refreshes, Finder metadata and orphan-record
cleanup. The final macOS bundle builds, and docs-check passes. No model math or
performance figure changes; the existing ledger remains actionable and unchanged.

2026-09-09. At the user's request, Delete session moved onto each gallery session's
timestamp row and the separate management panel was removed. Rows use the complete
session catalog, so empty sessions and older sessions outside the 200-image preview
limit remain accessible. Confirmation still receives the full image count. All 11
browser flows and the bundle build pass. Two local reviews and an external Qwen
review ran; a heading-margin regression was corrected. Qwen's suggestion that
session counts were capped at 200 was checked against the separate session catalog
and its existing 205-image test: only the image previews are capped. Docs-check
passes, and the ledger is unchanged.

## Image drops and diagnostic logging

2026-09-09. The Choose image controls were click-only despite their dashed borders.
Source and ControlNet fields now accept one dropped PNG/JPEG, including replacement
of a populated field. Tauri's webview drag events provide paths and physical
positions. The bridge converts positions to logical coordinates and tracks display
scale changes; hit testing respects modal overlays. Native paths use the existing
Rust decoder. Browser preview has a File fallback with signature-based MIME, the
same 100 MB/8192-pixel limits, and no URL fetching. Imports share functional settings
updates with the chooser, preserve concurrent prompt edits, and invalidate old masks
or control previews as appropriate. Late superseded imports are ignored.

The Rust logging plugin writes `~/.local/state/xwen/image-studio/logs/image-studio.log`
in a mode-0700 directory. It rotates at 5 MiB with one archive; the plugin's
`KeepSome(1)` means one archive plus the current file. Only application targets are
enabled. Startup, shutdown, command outcomes and a chained panic hook use the sink.
The frontend sends console messages, handled errors, uncaught exceptions, rejected
promises and React failures through a custom batched command. Handlers install before
the App module loads, and a React boundary supplies a reload screen.

The frontend queue holds at most 200 records and sends at most 50 per command.
Messages and sources are bounded, circular objects and throwing getters are handled,
and forwarding failure uses the original console to avoid recursion. Both sides
filter credentials and image data; configured and draft API keys are registered before
associated errors can be logged. Normal command events do not include request bodies.
Server settings exposes Show application log. Pending frontend records are best effort
on process termination; no crash-upload service is introduced.

Verification: TypeScript, 18 unit tests, 13 Rust tests and 16 browser flows pass.
The browser suite exercises drops, replacements, mask reset, modal shielding,
concurrent edits and forwarding of real console errors, uncaught exceptions and
rejected promises. The Rust test writes actual temporary logs, checks redaction,
rotation and command failures. The macOS bundle builds. A native startup attempt
was denied filesystem access by this execution environment before logging could
initialize; Finder drop delivery is therefore not claimed as manually verified.
Two local reviews and an external Qwen review ran; display scale changes, browser
MIME normalization and redaction ordering were corrected. The remaining external
claims were checked and refuted. Docs-check passes; model math and the ledger are
unchanged.

## Not taken now

Pending queue recovery across an app restart is not implemented. Completed results
and their input assets are durable, but pending work is in memory; reopen when
long-running unattended batches need interruption recovery. The current queue is
bounded at 1000 images and history reloads at 200 previews; larger collections can add
paging and lazy binary previews when those limits impede a real workspace.

The desktop bundle is a local development build. Distribution signing, notarization,
auto-update and forwarding new CLI arguments into an already running instance are
outside this local app request; reopen for distribution or single-instance use.
`open -n … --args <workspace>` explicitly starts another instance today.

This arc changes no model math or performance figures. No TODO item is closed by
it; the existing Front remains ranked, with its next experiments and prerequisites
already recorded. The optional client extensions above have no measured cost or
waiting user, so they stay in this record rather than entering the open ledger.

## External API sources

The implementation research checked the official
[Tauri Vite setup](https://v2.tauri.app/start/frontend/vite/),
[Rust command API](https://v2.tauri.app/develop/calling-rust/),
[dialog plugin](https://v2.tauri.app/plugin/dialog/),
[CSP configuration](https://v2.tauri.app/security/csp/),
[macOS application bundles](https://v2.tauri.app/distribute/macos-application-bundle/),
[reqwest API](https://docs.rs/reqwest/latest/reqwest/), and
[Playwright web-server setup](https://playwright.dev/docs/test-webserver).
Browser tests exercise the frontend and a mock bridge; they do not replace Rust
command tests or a native bundle launch.

The file-drop and logging follow-up checked the installed Tauri APIs against the
[webview drag event API](https://v2.tauri.app/reference/javascript/api/namespacewebview/#ondragdropevent),
[window scale API](https://v2.tauri.app/reference/javascript/api/namespacewindow/#scalefactor),
[logging guide](https://v2.tauri.app/plugin/logging/) and
[Rust logging builder](https://docs.rs/tauri-plugin-log/latest/tauri_plugin_log/struct.Builder.html).
