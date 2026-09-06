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

**For the dense `qwen3` graph the authority is llama.cpp `src/models/qwen3.cpp` and HF
`modeling_qwen3.py`, and here the two agree.** The Qwen 3.6 entry above warns that HF is
hazardous to diff against because of conversion-baked deltas; that warning is about
GGUF, and it does not transfer. We read the HF safetensors directly, so there is no
converter between us and the weights, no pre-baked `w+1` norm and no tiled V-head
permutation: GQA broadcast here is `repeat_interleave` semantics, KV head `j` serving Q
heads `4j..4j+3`, which is the form the 3.6 GGUF path calls WRONG for itself. Both
references were read rather than assumed and they agree on every point that could have
gone the other way: q_proj then per-head RMSNorm over [128] then rope, the same for k,
with v untouched by either; full NEoX rope over all 128 head dims (llama.cpp asserts
`n_embd_head == n_rot`), against 3.6's partial 64 of 256; scale `1/sqrt(128)`; plain
RMSNorm with no `+1`; SwiGLU; no biases; and with `use_sliding_window` false every layer
is full attention. The pinned oracle checkout already carries a `qwen3.cpp` byte-identical
to master, so no re-pin was needed. Two runtime facts are NOT in `config.json` and are
supplied by the architecture definition rather than defaulted: `NormVariant::Standard`
and `RopeSpec { head_dim, rotary_dim, theta }`, neither with a `Default` impl. That is
the same discipline as the removed `ssm_ba` fallback, inverted: a form that could have
gone the other way is stated once where it can be read, and a future variant has to be
implemented rather than assumed into existence (2026-09-06).

**The Stage 1 oracle for `qwen3` is per-position logits, and its CPU arm is not the
arithmetic being graded.** Neither existing oracle produces what a dense-4B gate needs:
`llama-eval-callback` dumps per-node sums with first-3/last-3 samples and computes
logits for the last position only, and `llama-perplexity --kl-divergence-base` stores
uint16-compressed log-probs with a 16-logit floor. `scripts/llama-logits-all.cpp` links
libllama, sets `batch.logits[i]` on every token, and streams raw f32 `[n_tokens,
n_vocab]` to disk with a sidecar recording backend, GPU layers, KV types, batch
geometry, flash-attn, threads, the GGUF sha256 and the llama.cpp commit; chunked decode
is ubatch-invariant, checked bitwise. The caveat is the load-bearing part. llama.cpp's
CPU path narrows F32 activations to BF16 before every BF16 matmul - the type traits set
`vec_dot_type = GGML_TYPE_BF16` and the llamafile fast path has no ARM branch, so the
conversion always runs - while its Metal path keeps F32 activations against BF16
weights, which is what xwen does. A CPU reference is therefore a different computation,
not a slower one, and the planned 2e-2 max-abs bar is provisional until the same binary
has produced the Metal arm under `--n-gpu-layers`. Two measurement definitions are fixed
now so that a later number cannot quietly mean something else: top-5 agreement is POOLED,
`sum |top5_candidate ∩ top5_ref| / (5 × positions)`, because per-position overlap of
five items moves in 20% steps and a per-position 99.9% is not a quantity; and the
encoder's relative error is per token, `max_i |x_i − r_i| / max(max_i |r_i|, 1e-6)`, the
denominator being that token's own largest reference magnitude rather than a global one
(2026-09-06).
