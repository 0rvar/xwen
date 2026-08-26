# Vendored qwen4exp reference (READ-ONLY reading material)

Per docs/qwen4exp-port.md D4: this is **not** an oracle. `reference/llama.cpp`
(pinned e9fa0781) remains the frozen oracle for the 3.6/3.8 parity cycles. These
files exist to be read, quoted and argued with — never built, never diffed against
as ground truth, never used to grade a parity tier.

## Source

- PR: ggml-org/llama.cpp **#27742** — "model: add Qwen3.8-Flash-Next (qwen4exp)",
  opened 2026-08-26 by `danielhanchen`.
- Head repo/branch: **unslothai/llama.cpp** @ `qwen4exp/qwen3.8-flash-next`
- Head commit: **bea3b12daee45876b0129a3602dc8f534ce30bf0**
- Base: ggml-org/llama.cpp `master` @ `4d19b287691e8f47fc303be420f630c40ec45684`
- PR size at this sha: 24 files, +2191 / -19.
- Fetched: **2026-08-26** (raw.githubusercontent + `gh api`).

## Status caveats

- **Unreviewed.** No approving review at fetch time; `mergeable_state: blocked`.
- The GitHub API reports `draft: false` at this sha (the port doc's "DRAFT PR"
  predates the PR leaving draft, or was never literally a GitHub draft). Treat it
  as unmerged and unreviewed either way — that is what matters.
- Qwen's promised **independent numeric check (JJJYmmm)** has not landed. The PR's
  own claim (ppl 4.0068) is self-reported and unverified.
- The branch is AI-drafted. Comments in it assert intent confidently; several
  assertions in `qwen4exp.cpp` are self-justifying comments, not tested facts.
- The PR may be force-pushed. Everything here is pinned to the sha above; if a
  file is re-fetched, re-pin the sha and re-check the deltas.

## Manifest

| file | upstream path | kind | sha256 |
|---|---|---|---|
| `qwen4exp.cpp` | `src/models/qwen4exp.cpp` | new file, verbatim | `7ba2db91581695cbd70f4b22e69cf86d6eeefd2c146c2eaa837b59ce0121ba72` |
| `qwen4exp.py` | `conversion/qwen4exp.py` | new file, verbatim (HF→GGUF converter) | `d927d6fd3dd22bcf83ed1188b8af6231d3ea9c0b92c8a8d477b63d989f2cf490` |
| `gguf-py.diff` | `conversion/__init__.py`, `gguf-py/gguf/{constants,gguf_writer,tensor_mapping}.py` | unified diff of the PR's hunks | `77fad3f11f25360719dc7d7a486dbdc4d81c16bbbcce4d153958c57ce4626133` |
| `llama-cpp-core.diff` | `src/llama-{arch,graph,hparams,kv-cache,memory-hybrid,model,model-loader,model-saver,quant}.{h,cpp}`, `src/models/models.h`, `tests/test-llama-archs.cpp` | unified diff of the PR's hunks (all C++ outside `qwen4exp.cpp`) | `43f80f52ede4d68fa8afe22128e47b5ba92dad75ac61edd512093ec9da024fe5` |
| `conversion-qwen-base.py` | `conversion/qwen.py` | **NOT part of the PR diff** — unchanged file at the same sha, vendored because `Qwen4ExpTextModel` is a thin subclass and the norm/+1, `-exp(A_log)` and V-head-reorder rules all live here | `56c65e7bc7817e624be3c5d9f6ca4a28e9e0ef89d4c6833d877bd86ecd798ecc` |

Re-fetch recipe:
`curl -L https://raw.githubusercontent.com/unslothai/llama.cpp/bea3b12daee45876b0129a3602dc8f534ce30bf0/<path>`
and `gh api repos/ggml-org/llama.cpp/pulls/27742 -H "Accept: application/vnd.github.v3.diff"`.
