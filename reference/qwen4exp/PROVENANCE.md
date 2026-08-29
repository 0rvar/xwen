# Vendored qwen4exp reference (READ-ONLY reading material)

Per docs/qwen4exp-port.md D4: this is **not** an oracle. `reference/llama.cpp`
(pinned e9fa0781) remains the frozen oracle for the 3.6/3.8 parity cycles. These
files exist to be read, quoted and argued with — never built, never diffed against
as ground truth, never used to grade a parity tier.

## Source

- PR: ggml-org/llama.cpp **#27742** — "model: add Qwen3.8-Flash-Next (qwen4exp)",
  opened 2026-08-26 by `danielhanchen`. **MERGED 2026-08-27** into
  ggml-org/llama.cpp `master` as squash commit
  **`6c84c7d5d8833c6e0df69628f75a0f599797934e`**.
- Pre-PR base (the squash commit's parent, i.e. the PR's merge-base with master):
  **`6fdd0ac8907fd973a42b876357823ad2124cd8ed`**.
- Follow-up: **`6fe74980162af0ed5e559870d5deccafaa034e7c`** (PR #27880,
  2026-08-28) — "model: qwen4exp: reduce number of graph splits". Touches only
  `src/models/qwen4exp.cpp` and `src/models/models.h`.
- **Pinned sha for everything below: `6fe749801`** (upstream `master`, not a fork).
- Merged PR size: 28 files, +2881 / -38 (the pre-merge sha was 24 files, +2191 / -19).
- Fetched: **2026-08-29**, from a `--filter=blob:none` clone of
  ggml-org/llama.cpp (not from the unslothai fork, and not via the PR diff API —
  the PR is merged, so upstream master is the source now).

## Status caveats

- **Merged, but the review it got is not a numeric verification.** Qwen's promised
  independent numeric check (JJJYmmm) still has not landed. The PR's ppl claim
  (4.0068) remains self-reported.
- The branch was AI-drafted and the merge did not rewrite it. Comments still assert
  intent confidently; several assertions in `qwen4exp.cpp` are self-justifying
  comments, not tested facts.
- **There is a substantial unmerged follow-up in flight.** Branch `origin/tmp-q4`,
  head `f91123d2d02907c665cee42d6428eee5ee35d356` (2026-08-28, Thiago Padilha,
  ~1450 lines) reworks QSA block selection ("build QSA blocks per sequence in
  token order and select complete blocks before expanding them to cache cells…
  rotate pooled keys with the first token's full M-RoPE position"), adds
  independent PLE embedding widths, metadata validation, indexer-cache updates
  after sequence copies, and recurrent-state rollback. **It is not vendored here**
  and it would change QSA numerics. Re-check before treating the vendored QSA
  selection as settled.
- The previous vendored snapshot was the pre-merge fork sha
  `bea3b12daee45876b0129a3602dc8f534ce30bf0`; anything in the port doc written
  against it is ~60 commits stale.

## Manifest

| file | upstream path | kind | sha256 |
|---|---|---|---|
| `qwen4exp.cpp` | `src/models/qwen4exp.cpp` | new file, verbatim @ `6fe749801` | `1fe800a11543d2af7aef898f75e6ee6ffb1f27f1486b4810cb926d94fc77e148` |
| `qwen4exp.py` | `conversion/qwen4exp.py` | new file, verbatim @ `6fe749801` (HF→GGUF converter) | `12a0a5aea7877fbb8fe35af041a9c34f8b57b05278871b22c24c650b9760dfc3` |
| `gguf-py.diff` | `conversion/{__init__,base}.py`, `gguf-py/gguf/{constants,gguf_writer,lazy,tensor_mapping}.py` | unified diff of the merged PR's hunks | `9ab1af23b4bb6244e153a82b4dbe81610060d9e9672fb97299806ba98c71b20a` |
| `llama-cpp-core.diff` | `src/CMakeLists.txt`, `src/llama-{arch,context,hparams,kv-cache,memory-hybrid-idx,memory-recurrent,model-loader,model-saver,model,quant}.{h,cpp}`, `src/models/models.h`, `tests/test-llama-archs.cpp` | unified diff, two sections (merged PR, then the graph-split follow-up); all C++ outside `qwen4exp.cpp` | `05e887b46d5dfdbab0f3870b315bf328ab62d8d85de162d715f22191be2a8af9` |
| `conversion-qwen-base.py` | `conversion/qwen.py` | **NOT part of the PR diff** — unchanged file at the same sha, vendored because `Qwen4ExpTextModel` is a thin subclass and the norm/+1, `-exp(A_log)` and V-head-reorder rules all live here | `3e3ea6be268c65915f8f5b8419edc15c291f0adf374b97216f436191d23456b4` |

Notes on what moved between the two vendored generations:

- `src/llama-memory-hybrid.cpp` is **no longer patched**; the merged PR adds a new
  `src/llama-memory-hybrid-idx.{h,cpp}` instead.
- `src/llama-graph.{h,cpp}` is **no longer patched at all**. `build_attn`'s `top_k`
  parameter and `build_attn_mask_top_k` are gone; the sparse mask build is now
  arch-local in `qwen4exp.cpp::build_attn_qsa`.
- `gguf-py/gguf/lazy.py`, `conversion/base.py`, `src/llama-context.cpp` and
  `src/llama-model.h` are new to the diff.
- `conversion/qwen.py` is still untouched by this PR (its diff against the old
  vendored copy comes from the unrelated `ca3d5a3e1` DSpark/Nemotron3.5 commit).

## Re-fetch recipe

```
git clone --filter=blob:none https://github.com/ggml-org/llama.cpp llamacpp-master
cd llamacpp-master && git checkout 6fe749801

git show 6fe749801:src/models/qwen4exp.cpp  > qwen4exp.cpp
git show 6fe749801:conversion/qwen4exp.py   > qwen4exp.py
git show 6fe749801:conversion/qwen.py       > conversion-qwen-base.py

# gguf-py.diff: the merged PR's converter/gguf-py hunks (the follow-up touches none)
git show --format='' 6c84c7d5d -- conversion/__init__.py conversion/base.py \
    conversion/qwen.py gguf-py/gguf/constants.py gguf-py/gguf/gguf_writer.py \
    gguf-py/gguf/lazy.py gguf-py/gguf/tensor_mapping.py

# llama-cpp-core.diff: section 1 = merged PR, section 2 = the follow-up
git show --format='' 6c84c7d5d -- src/CMakeLists.txt src/llama-arch.h src/llama-arch.cpp \
    src/llama-context.cpp src/llama-hparams.h src/llama-hparams.cpp src/llama-kv-cache.h \
    src/llama-kv-cache.cpp src/llama-memory-hybrid-idx.h src/llama-memory-hybrid-idx.cpp \
    src/llama-memory-recurrent.h src/llama-memory-recurrent.cpp src/llama-model-loader.cpp \
    src/llama-model-saver.h src/llama-model-saver.cpp src/llama-model.h src/llama-model.cpp \
    src/llama-quant.cpp src/models/models.h tests/test-llama-archs.cpp
git show --format='' 6fe749801 -- src/models/models.h
```

The two core sections are concatenated rather than taken as one
`6fdd0ac89..6fe749801` range diff, because 17 unrelated master commits land in that
range and several of them (`ca3d5a3e1` DSpark, `866322481` GDN/LID op gating,
`18443257a` ctx-per-slot, `4e97ac86e` test-save-load-state) touch the same files.
`git show` per qwen4exp commit keeps the diff to the qwen4exp additions only.

Note: the shell here is zsh, which does not word-split unquoted variables — pass
those pathspecs literally, not through a `$VAR`.
