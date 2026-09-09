# Memory ownership and the September 9 freeze

## Incident evidence

On 2026-09-09, Orvar reported concurrent Flash-Next and image generation, rapidly
rising memory pressure, a frozen desktop and a reboot. The supplied crash log dates
the panic to 10:58:59 CEST. It reports a 65-second termination timeout in
AppleCentauriManager, which the installed driver's device registry identifies as
managing Wi-Fi/Bluetooth firmware. Compressor limits and swap space were reported OK.
The log establishes a driver panic; it does not establish memory pressure as its cause.

The application hazard was already explicit in
[the pipeline record](zimage-pipeline.md#not-taken-now-arc-c): independently resident
Flash-Next and images thrash. The default Flash-Next file has 82.53 GB of trunk weights
and a 28.80 GB demand-paged PLE table; images have roughly 20 GB of final weights plus
temporaries. Those are file/allocation estimates, not a sum of measured physical use.
No available telemetry captured the incident's actual peak or crashing request.

## Implementation

`src/memory.rs` owns one resident inference slot per user. Separate threads coordinate
locally; separate processes lock the same owner file under a private directory in
`/private/tmp`. A second file carries waiting intent. Idle engines poll this intent
even with idle unload disabled. A lease covers loading, active work and retained
models. Teardown drops host slots and tensors and drains the Metal device before
releasing ownership. Failed GPU drains terminate the process rather than admitting
another load. Load guards cover partial construction and unwinding.

Ordinary contention finishes the current request, then evicts. It does not make a new
host snapshot during eviction. Existing disk writes may still own buffers; they are
not cancelled. The server's language request watchdog includes waiting for ownership.
Image ownership waits are bounded by the server queue timeout and client lifetime.
CLI commands can be interrupted normally while waiting. Metadata, fetching and stats
do not take ownership. Direct library callers must acquire and retain a lease themselves;
older binaries and other applications do not participate.

Admission samples physical RAM, anonymous/wired/compressor pages, process footprint
and pressure. It adds projected new allocations to system usage and preserves
max(16 GiB, 10% RAM). It does not subtract process footprint from a different OS
accounting ledger. Missing required RAM/use readings refuse admission; unknown pressure
is recorded as unknown. Metal's recommended working-set size is diagnostic, not free
RAM. Language admission includes initial KV/state, an 8 GiB scratch allowance and up
to 4 GiB of demand-paged PLE working space; KV reallocations and server host-cache
growth get separate admission checks. PLE's full mapped size is not assumed resident.

Images reserve 40 GiB, another 24 GiB for control and 8 GiB per LoRA. These are
conservative allowances, not measured peaks. The output envelope is at most 1,048,576
pixels, alongside the existing geometry rules. Preprocessing cannot consume the
reservation for a later encoder/transformer load. Warm jobs reserve temporary allocations
on every request, crediting only a 16 GiB lower bound on retained model weights.
Switching adapters releases the old pipeline
before admitting and loading the replacement.

The host monitor samples every 500 ms. Warning pressure stops new admissions and
causes idle owners to unload. Critical pressure stops language work at its cancellation
boundaries and image work between transformer blocks, denoising steps and VAE phases.
Reaching the system headroom limit also stops work, including when native pressure
notifications are unavailable. The fallback uses cached measurements at runtime.
Client disconnect and shutdown use the same image checks. These checks prevent future
GPU submissions; they cannot preempt an already-running command or repair a hung driver.
No arithmetic or per-block synchronization was added.

`~/.local/state/xwen/memory.jsonl` records pressure transitions, periodic active-process
readings, admission decisions and allocation events. Events include estimates separately
from measurements; device events include Metal allocations and its recommended size.
The log contains no prompts. Writes coordinate across processes and clear the file at
16 MiB while keeping the inode stable. An unavailable telemetry file is reported.

## Verification

Tests exercise exclusive threaded and cross-process ownership, cancelled waiters,
lease release, memory accounting, missing measurements, pressure refusal, bounded
telemetry and image-area limits without loading checkpoints. Server regressions cover
idle deadlines, waiting watchdogs, pressure errors and a stalled client. Image
cancellation tests use invalid tensors and nonexistent files and require cancellation
before any tensor operation or file open. No exhaustion test was run on this machine.

Review caught and fixed a watchdog bypass while acquiring ownership, speculative-page
accounting, ignored GPU drain errors and preprocessing credit against an unallocated
image pipeline. Independent agents reviewed the ownership/pressure and image paths.
The outside-model reviewer was unavailable: `qwen --ping` reported the local server
offline. Final checks passed: 501 server tests, 11 memory tests (including a child
process), eight cancellation-filter tests, the KV admission growth regression and
15 CLI tests. Some filters overlap. `cargo check --all-targets`, `cargo build --release`,
formatting, diff checks and `bun scripts/docs-check.ts` passed. The release binary
refused a 1536x1024 image before loading a model. Telemetry file creation was denied
inside the sandbox and reported visibly; bounded writer tests used a temporary file.

## Not taken now

- Reproduce the reboot by exhausting memory: unnecessary to establish the missing
  ownership guard and unsafe on the user's workstation. Reopen only with an isolated
  machine and a recovery plan. The wireless-driver causal question remains unresolved.
- Concurrent residency of smaller checkpoints: serialize initially. Reopen after
  measured combined peaks justify explicit reservations, including transient loads.
- Raise the image area limit or tighten peak allowances: measure one isolated pipeline
  first, from cold load through encoder, denoise and VAE, with control/LoRA variants
  measured separately. The existing footprint ledger item remains open.
- Preempt a single GPU command, preprocessor invocation or weight-loading stage:
  cooperative checks cannot do this. Reopen if telemetry shows a stage outlasting the
  pressure response window; split that stage or isolate it in a supervised worker.

## Next experiment

The existing image-footprint item now has telemetry to use. With other inference
processes stopped, run one updated `xwen image` at 512 squared, then 1024 squared,
collecting `memory.jsonl` and a user-shell footprint trace. Compare cold-load peaks,
warm render peaks and post-unload readings before trying a control variant. Stop on
admission refusal or warning pressure; do not relax a guard to complete the experiment.
This prices the 40/24/8 GiB allowances and is a prerequisite to raising image area or
allowing co-residency. No measured performance or image-footprint figure changed here.
