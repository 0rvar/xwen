# Server activity and prompt accounting

## Anthropic prompt usage

2026-09-10. The Anthropic adapter reported the full prompt in `input_tokens` and
again across the two cache fields. Clients adding the three fields therefore saw
twice the prompt size. The adapter now reports disjoint buckets: zero uncached
input, the reused prefix as cache reads, and the prefilled remainder as cache
creation. Both buffered responses and streaming starts use the same helper.
The token-count endpoint still returns the full prompt. See the
[accounting decision](../decisions/serving.md).

The regression covers cold, partial and full cache hits in both response modes.
Its sum assertion failed with 200 against 100 before the change. All 61 Anthropic
tests passed after it. Streaming still announces planned creation before prefill
finishes; the API split is not a measurement of completed work on an interrupted
request.

## Prefill timing

2026-09-10. The history numerator already counted forwarded tokens, excluding
cache reads. Its denominator measured the asynchronous `prefill_tokens` call,
so a snapshot readback or the first decode could absorb outstanding prefill work.
This explains how a short uncached tail could appear faster than its GPU execution;
the size of that distortion has not been benchmarked.

Serve now drains pending restoration before opening a prefill span's timer and
drains submitted work before closing it. Spans end at the existing snapshot or
prompt boundaries. Chunks within a span remain pipelined. Cancellation and forward
errors also close the interval. The TUI receives cumulative forwarded tokens and
completed phase time separately from prompt-progress ticks; event arrival timing
no longer sets the prefill rate. The same completed measurements reach metrics.

Batch rows show forwarded and decoded counts, item outcomes and total duration.
Their per-phase rates are omitted: the shared generator's cumulative batch timing
still measures asynchronous calls. Changing that accounting is not taken now;
reopen it when batch phase rates are needed, starting at `Generator::prefill_spend`
and the delta read in `batch::run`. Preserve chunk pipelining when placing completion
boundaries. No new performance figure replaces the measured baselines.

## Mixed activity

2026-09-10. The header leads with the resident checkpoint and current slot context.
The default checkpoint's file size no longer labels a different resident model.
Every successful language-model load supplies its resolved context limit; moving
away from a smaller-window model can restore the larger limit. Image residency
uses the image worker's shared atomic flag, so event delay cannot keep an unloaded
pipeline in the header. Diffusion has no language slot context. During language
dispatch the request's slot stays unknown until cache resolution identifies it;
while idle, the header shows the retained live slot.

Queue and history use time, API, model, outcome and inference statistics. Text rows
show cache/new/output counts and completed prefill/decode measurements. Diffusion
rows show dimensions, image counts, actual executed steps and render timings.
Preprocessing is a distinct activity. The two workers keep independent active
state and contribute to one bounded chronological history.

Image jobs carry a trace from submission to disposal, so queue refusal, disconnect,
shutdown and errors all remove their queue entries. Caption timing drains at its
phase boundary. Denoising and VAE times come from the pipeline's existing completed
measurements. Progress advances per completed image. A pipeline failure partway
through an image cannot return its phase timings, so partial-job phase totals cover
only completed pipeline runs. Per-step progress and partial-image timing are not
taken now; reopen them if diagnosing a stalled or failed individual image needs
finer detail, at `ZImagePipeline::generate_cancellable`.

## Verification

The focused debug-profile suites passed: Anthropic 61, TUI 38, image handlers and
lifecycle 18, queue 21, log 22, and the cached-tail accounting regression 1. The
accounting test verifies propagation into history/metrics; GPU completion follows
the explicit device barriers and was reviewed in code, not priced in a benchmark.
TestBackend renders at 100x35 and 80x30 show both image and text statistics without
truncation. No model math changed, so no parity gate was run.
`cargo check --bins`, formatting and `bun scripts/docs-check.ts` also pass.

Independent reviews covered Anthropic accounting, timing and backend lifecycle.
Display review caught delayed image-residency state and the previous slot leaking
into a new request's dispatch header; both now have regression coverage. Cancelled
image jobs retain that outcome even when their terminal record also carries an
explanatory error.
The outside Qwen review was attempted but lost its connection to the local server;
the subsequent reachability probe failed. No Qwen approval is claimed.
