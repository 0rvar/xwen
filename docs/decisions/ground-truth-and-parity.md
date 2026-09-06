# Ground truth and parity methodology

One topic of [docs/decisions.md](../decisions.md), the index of decisions by topic; moved
here verbatim on 2026-09-06. Dated paragraphs, newest additions appended within their topic.


**Upstream llama.cpp master replaces the poolside fork as parity ground truth.** The
reference implementation is `src/models/qwen35.cpp`, `src/models/qwen35moe.cpp`, and the
shared `src/models/delta-net-base.cpp` (llama.cpp was refactored; per-arch graphs no
longer live in llama-model.cpp), running the identical ggml-org GGUF. The three-tier
gate methodology (strict oracle cosine / fork-equivalence / greedy replay + perplexity
bound) carries over from laguna; floors must be recalibrated for the Qwen checkpoint
quant mix and are not laguna's numbers (2026-07-28).

**The oracle is PINNED to a commit, not tracked.** `reference/llama.cpp` must be a
checkout at the exact sha recorded in docs/parity.md. A different oracle build can move
the achieved cosines, which would silently invalidate the floors calibrated against it,
so moving the pin is a deliberate act paired with a re-calibration — never a `git
pull`. The harness itself is indifferent to HOW the checkout is materialized; what it
requires is that the sha be recorded and deliberate.

**The oracle checkout is a git SUBMODULE** (shallow, declared in .gitmodules), with
only `reference/llama.cpp/build` ignored. Resolved by the owner the same night it
arose: the P7 session had materialized it as a plain gitignored clone while a
submodule gitlink was staged concurrently; the owner picked the submodule, whose
gitlink makes the oracle sha reviewable in the diff — moving the pin is a staged
change the owner approves, which is exactly the "deliberate act" the paragraph above
requires (2026-07-28).

**Floors are calibrated across BOTH checkpoints and all fixtures, then set under the
WORST observed value.** The gate constants are global (one `COS_MIN_MM` for whatever
file is gated), so a floor derived from one checkpoint or one prompt would be a
coin-flip on the others. The measured spread is in docs/parity.md "Floors": the
strict floor sits ~1e-4 under the worst classic-path value and the mm floor ~5.4e-4
under its worst, roughly 1.7x the observed prompt-to-prompt spread. Per-checkpoint
applicability remains enforced by the procedure, not by code — the ledger item for
binding a floor to a checkpoint hash is still open (2026-07-28).

**Tap-name translation lives in the harness, not in the engine.** Our tap names and
llama.cpp's `cb()` node names disagree on three of nine taps, and the mixer output has
two different llama.cpp names depending on layer kind. `scripts/parity.ts` owns the
mapping (`refTapNames`) rather than renaming engine taps, so the parity harness can
follow upstream's naming churn without touching model code — and so a mapping mistake
is a harness bug, not a silent change to what the engine records (2026-07-28).

**HF transformers is the secondary reference, with two conversion deltas that make
diffing against it hazardous:** GGUF norm weights are pre-baked `w+1` (HF Qwen3.5 norms
are Gemma-style zero-centered `(1+w)`) for every norm EXCEPT the DeltaNet `ssm_norm`,
and GGUF V-head ordering is tiled (llama.cpp permutes V-side weights at conversion to
suit `ggml_repeat`) where HF safetensors are grouped (`repeat_interleave`). We read
GGUF, so we use the pre-baked/tiled forms directly and never "fix" them (2026-07-28).
