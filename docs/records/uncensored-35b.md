# HauhauCS uncensored 35B integration

2026-09-09. The owner requested HauhauCS's aggressive Qwen3.5-35B-A3B Q4_K_M
checkpoint under `Qwen3.6-35B-A3B-uncensored`, plus a download command and GUI
preference over the regular 35B.

## Checkpoint and identity

The API alias is deliberate; these are Qwen3.5 fine-tuned weights, not a Qwen3.6
release. Source:
[HauhauCS repository](https://huggingface.co/HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive).
The file is `Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf`,
21,169,117,248 bytes. Upstream revision at verification was
`f34c1414c2f9a3629187098bdf0e90c85f5034f5`, with SHA256
`c117a47c5d8d1bb91d68031aaa77891f10118338e1174accc48c55ee3fff8717`.

The GGUF header declares `qwen35moe`, with the same 40-layer graph, 2048 hidden
width, 256 experts, top 8, and attention/DeltaNet geometry as the existing 35B.
All 248070 real token IDs and the merge table match `reference/tokenizer.json`.
Its embedded chat template matches the base Qwen3.5 template. The existing Qwen36
chat dialect and 35B sampling defaults apply. No model math changed.

This Q4_K_M mix differs from the regular registry checkpoint: 301 F32, 355 Q4_K,
30 Q5_K and 47 Q6_K tensors, with no Q8_0. QKV projections are Q5_K; half the
expert down planes are Q6_K. Existing general and indexed matmul paths support
these types. No speed comparison was run.

Identification recognizes both the upstream full name and the API alias.
A base-name occurrence contained within the uncensored alias belongs to that
longer name. Separate occurrences naming different checkpoints remain ambiguous.
The file's identity still cross-checks an explicit `--model-size`.

## Download and selection

`xwen fetch-model Qwen3.6-35B-A3B-uncensored` downloads through `ensure_model`
and `ensure_file`. The latter already serves Hugging Face LoRA downloads, including
resume and cache locking. `35b-uncensored` is the CLI alias. The new command accepts
all registry names and aliases and fetches model files only; the existing `fetch`
command retains its drafter behavior.

The registry entry has no drafter and `auto_fetch` is false. The existing shared
availability predicate controls both listing and request admission. An incomplete
download is not available. Image Studio's prompt and chat tools query `/v1/models`
on each request and prefer the exact uncensored ID, falling back to the regular
`Qwen3.6-35B-A3B`. Both calls use one captured server configuration for discovery
and completion. The server's own default remains Flash-Next.

## Verification

Backend checks passed: 32 hub tests, 35 serve tests, 72 configuration tests and
the new CLI parsing test. The release build and formatting check passed.
GUI checks passed: 18 Rust tests (one existing live-image test ignored),
21 Bun tests, typecheck and frontend production build. The HTTP fixture exercises
both tools, both model choices, refreshed availability, authentication and a
similarly named ID that must not match.

The browser suite passed 24 tests and failed one existing selector at
`image-studio/e2e/studio.spec.ts:171`: it asks for `Restore settings`, while
the committed `Gallery.tsx` renders `Restore selected`. Both files are unchanged
by this work. This unrelated test repair was not taken now; revisit it when
working on gallery restore coverage. Failure output was preserved under
`/tmp/uncensored-gui-playwright-report-1788985054311` and
`/tmp/uncensored-gui-test-results-1788985054311`.

The live server returned no uncensored entry while the download was incomplete,
and an OpenAI chat request naming it returned HTTP 400 with an explicit fetch
instruction. Independent backend and GUI reviews found no actionable defects.

The outside-model review first exceeded its context limit, then completed against
the staged diff. Its four concerns were checked: malformed `/v1/models` responses
are errors rather than evidence of absence; per-request discovery deliberately
refreshes availability; the production predicate includes both `servable` and
`auto_fetch`, contrary to the review's description; and the generic error text
is in the browser preview mock, not the native server error path. No changes were
needed. Review output is `/tmp/uncensored-qwen-review-small.log`.

The macOS application bundle built successfully. `cargo install --path . --locked`
installed the updated CLI. Full-file download and generation verification follow
once the ongoing `fetch-model` transfer completes.
