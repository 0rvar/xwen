# qwen4exp reference — history, not material

**The source of truth is the `reference/llama.cpp` submodule, pinned at
`6fe74980162af0ed5e559870d5deccafaa034e7c`.** Read qwen4exp upstream there:
`reference/llama.cpp/src/models/qwen4exp.cpp` (graph),
`reference/llama.cpp/conversion/qwen4exp.py` (HF→GGUF converter),
`reference/llama.cpp/conversion/qwen.py` (the Qwen3Next base it subclasses — the
+1 norm folding, `-exp(A_log)` and V-head reorder live there), and
`reference/llama.cpp/gguf-py/gguf/` (constants, tensor_mapping, lazy, writer).
Nothing is vendored into this directory any more.

## How it got here

- **2026-08-26**: PR #27742 ("model: add Qwen3.8-Flash-Next (qwen4exp)") was
  still unmerged, so five files/diffs were vendored read-only at the fork sha
  `bea3b12daee45876b0129a3602dc8f534ce30bf0`.
- **2026-08-27**: the PR merged into ggml-org/llama.cpp master as squash
  `6c84c7d5d8833c6e0df69628f75a0f599797934e` (pre-PR base
  `6fdd0ac8907fd973a42b876357823ad2124cd8ed`); follow-up `6fe749801` (#27880,
  "reduce number of graph splits") landed 2026-08-28.
- **2026-08-29**: re-vendored at `6fe749801`, then the vendored copies were
  deleted outright — the submodule was bumped e9fa0781 → `6fe749801` instead, so
  there is ONE oracle for all checkpoints and no second copy to drift.
  The 3.6/3.8 parity gate must be re-run against the new pin before it is
  trusted; docs/parity.md tracks that.

`UPSTREAM-DIFF-2026-08-29.md` is kept beside this file: the semantic reading of
`bea3b12d` → `6fe749801`, written while both existed. It is history — a
point-in-time analysis, not a description of anything checked in here.
