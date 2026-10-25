# Flint correctness: the graph, the vocabulary and the oracle

The record of the arc that took Flint (`ggml-org/Flint-GGUF`, Q4_K_M, arch `flint`) from a
model card to a checkpoint that is servable on every language surface, generates end to
end from `xwen generate` and passes a parity gate against llama.cpp, 2026-10-25. Serve and
batch are covered by unit tests, not by a run. The oracle pin landed first as b767234 and the code
as ca415f8. Decisions are in
[docs/decisions/flint.md](../decisions/flint.md), the parity track in
[docs/parity.md](../parity.md) "The Flint track", and the architecture and traps in
AGENTS.md's Flint section. This file holds the research, the tables, the brackets, the
review round and what was not taken.

## What it is

[RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint): a text-only MoE for coding
agents, Apache-2.0, trained from scratch by Recursive, 39,785,287,456 parameters, about
2.5B active per token. The code skeleton is Qwen3-MoE, so the tensor names are, but the
weights, the vocabulary (150,144) and the chat template are its own. ggml-org's Q4_K_M is
23.39 GB: routed experts and the dense lead FFN Q4_K, attention, shared expert, embedding
and head Q8_0, norms, routers and sinks F32.

Shape: 49 layers, hidden 2048, 32 Q and 4 KV heads of 128. 35 layers are sliding-window
(513 keys including the current one) with NEoX rope over all 128 dims at theta 1e4; the 14
others (0, 1, 5, 9, ..., 45, 48) are full attention with no positional encoding. Every
layer has QK-RMSNorm before rope and a learned sink per Q head. Layers 0 and 1 are a dense
SwiGLU of 6144; the other 47 are MoE, 256 experts of 512, sigmoid scores, top-6,
renormalized, plus an ungated shared expert of 1024.

Upstream support landed the same week: llama.cpp (#30928, merged 2026-10-21 as a37cc461,
plus #30861's down-projection rescale), transformers main, vLLM and mlx-lm.

## How it was built

Two forks in parallel on disjoint files, then one fix fork after the review round.

The graph runs inside the trunk's layer loop as `Mixer::Flint(FlintAttn)`
(decisions/flint.md "Flint's attention is a `Mixer` variant"). Semantics were settled from
the GGUF's own metadata and from `src/models/flint.cpp` at a37cc461, not from the research
summary:

- `sliding_window_pattern` and `rope_pattern` are identical bool[49] arrays, `true` meaning
  a sliding layer and a roped layer.
- The pre-FFN norm tensor is `ffn_norm`; the Qwen files call it `post_attention_norm`.
- Sinks are `attn_sinks.weight` [32] F32, raw and unscaled in the softmax denominator
  (`ggml_soft_max_add_sinks`).
- SWA is `LLAMA_SWA_TYPE_STANDARD`, so the window counts the current position.
- KV head is `h / 8`, by ggml broadcast.
- The MoE clamps the renormalizing sum at 6.1e-5, as the 35B does.
- There is no expert-selection bias tensor and no `rope.dimension_count`.

`XwenConfig::flint_from_gguf` refuses a gating other than sigmoid, `expert_weights_norm`
false, a weights scale other than 1, an expert-selection bias, and partial rope, since each
would run and return wrong numbers.

The ring cache `LayerCache::Swa` had existed since the laguna fork with its rollback,
snapshot and disk framing tested, but nothing built one. `LayerKind::Swa { window }`
constructs it now. `CacheGeometry` counts the 14 full layers for KV (28,672 B per token)
and adds the rings to a snapshot (36.8 MB).

The tokenizer fork generalized `Specials` so a vocabulary that is not ChatML resolves
(the layout is read off the vocabulary; Qwen's resolves exactly as before), embedded
Flint's `tokenizer.json`, added `VocabFamily::Flint` and wrote `ChatDialect::Flint` from
the jinja.

## Tokenizer and template, against llama.cpp

`tests/fixtures/flint-prompts.json`: 25 prompts, longest 3654 tokens, ids from
`llama-tokenize` of the Flint checkout. The embedded tokenizer matches all 25, and a test
holds it token-for-token equal to the GGUF's `tokenizer.ggml.tokens`. Flint's tokenizer
has no normalizer, so the NFC caveat of the Qwen fixtures does not apply.

`tests/fixtures/flint-chat.json`: 14 conversations, byte-exact through llama-server's
`/apply-template` on both the GGUF's embedded template and the vendored copy, 13 of 13
per arm with one skipped. The skipped case is the developer turn: llama-server rewrites
`developer` to `system` before the template runs, so the endpoint cannot render the
template's own `<|developer|>` branch, and that case is graded by the Rust tests against
the jinja. The oracle caught one mistake: HF and llama.cpp render with `trim_blocks`, so
tool schemas sit between `<tools>` and `</tools>` with no newlines.

## Parity

The fixture is llama.cpp Metal at a37cc461, top-10 ids and logits at every position of
seven prompts, 2562 positions, 423 KB. llama.cpp's two backends disagree with each other
on this model far more than on the dense 4B, and that set the bars:

| comparison | mean \|Δ\| | top-5 | argmax |
| --- | --- | --- | --- |
| llama.cpp CPU vs Metal | 0.613 | 0.863 | 0.907 |
| xwen vs Metal, prefill chunk 512 | 0.210 | 0.949 | 0.970 |
| xwen vs Metal, prefill chunk 2048 | 0.203 | 0.947 | 0.972 |
| xwen vs Metal, decode 300 to 602, across the ring | 0.238 | 0.921 | 0.940 |
| wrong graph: rope on every layer | 2.84 | 0.555 | 0.642 |
| wrong graph: sinks dropped | 11.98 | 0.080 | 0.047 |
| wrong graph: window 512 | 0.214 | 0.945 | 0.969 |

The bracket columns are the 148-position run the graph fork measured before the brackets
moved to every prompt, except the window-512 row, which is every prompt at the shipped
chunk.

Why the backends disagree, traced per layer on both llama.cpp backends: on one row their
per-layer drift is 4 to 16%, sigmoid routers flip near-tie experts (the sixth-ranked
expert at layer 8 was 0.078 against 0.075), and at layer 46 the two backends pick
different experts with massive activations. The evidence that xwen's graph is right is
that layer 2's routing ids and weights and its routed and shared outputs match the
eval-callback trace to every printed digit, that the per-layer drift grows smoothly with
no jump at any layer kind, and that turning on every f32 fallback leaves the numbers
unchanged. One row stays open: on the last token of `parity-code-short`, at layer 46,
both llama.cpp backends pick expert 193 and xwen does not. It reads like the same chaos;
it rests on one traced row.

The worst-window bar grades 64-position windows (argmax 0.82, top-5 0.82, mean 0.75).
xwen reads 0.922 / 0.894 / 0.638 at prefill and 0.891 / 0.866 / 0.445 at decode; the two
asserted brackets read 0.17 / 0.28 / 11.8 or worse. The worst single position is 18.0 off,
reported and not gated.

## The f16 overflow on the down projection

Measured on `long-1400`, prefill chunk 512. The input to the routed experts' down
projection reaches **297,857**, 4.5x f16's 65,504; the shared expert's reaches 6,997 and
the dense lead layers' 3.35. The default prefill `mm_id` stages that input as half, the
existing per-column rescale (`needs_rescale`, scaled to 32768) triggers on it, and the gate
passes on that path. Decode reads f32. The shared expert and the dense layers go through
the vendored dense gemm, which does not stage the activation to half; a test drives
activations of 1e6 through it at Q8_0 and Q4_K and stays finite and within 1e-3 of
QMatMul.

## The prefill precision gap

Found by the fix fork while building the chunk test, and not fixed. Each layer was
recomputed in f64 on the CPU from the dequantized weights and that run's own tapped input:

| layer | prefill, mean / worst per-row rel L2 | single-row decode |
| --- | --- | --- |
| 0, dense SwiGLU | 3.7e-2 / 5.2e-2 | 1.9e-7 |
| 2, MoE | 2.4e-2 / 4.2e-2 | 1.8e-7 |
| 2, routed experts forced to the per-row gemv | 7.7e-4 | |

f32 activation tiles (`XWEN_MM_ID_CLASSIC`) change nothing, so it is neither the
activation nor overflow. It is the weight tile staged as f16, amplified by cancellation in
the down projections, and it is the same on the vendored dense gemm and on candle's
QMatMul. llama.cpp Metal's `mul_mm` stages weights the same way, so the oracle carries it
too, which plausibly explains part of llama.cpp's own CPU-vs-Metal spread. The probe is
`/tmp/flint-fix-probes/flint_mlp_probe.rs`, outside the tree. TODO.md holds the item.

## Memory

Peak footprint of `xwen generate --model flint` beyond the mapped weights, production
attention with the 256 MiB tile cap:

| prompt tokens | peak |
| --- | --- |
| 9,895 | 10.97 GB |
| 28,717 | 16.28 GB |
| 56,883 | 31.89 GB |

Before the tile cap the 57k row read 41.3 GB and the 29k row 22.1. Growth is superlinear
and draining between chunks barely moves it (16.28 to 15.94 GB with `XWEN_CHUNK_SYNC`), so
something that scales with keys is retained per forward, most likely the per-layer f32 K/V
copies and the mask. `flint_scratch_bytes` is a quadratic through the three points plus
5%, and admission prices it at KV growth rather than at load, so a short prompt loads as
before and a long one is refused or waits. The fit extrapolates to ~115 GB at 131k, past
this machine. The kernel arc is expected to retire the f32 copies; until then a Flint
prompt past about 60k tokens is refused.

## Timing observations, which are not figures

`lowpowermode 1`, an unpinned dev build, the f32 attention: decode 47.7 tok/s on a
24-token prompt, 23.7 at a 3.6k context, prefill 292 tok/s at 3654 tokens. Decode halves
with context because the f32 attention widens the full layers' whole f16 cache to f32 on
every step. The fused MoE glue is also off for Flint. None of these goes in perf-state.md.

## The review round

Five reviewers on the uncommitted tree: two narrow ones on the cheaper tier (model math;
tokenizer, chat and serve), one architectural, Codex and DeepSeek. Qwen ran no review,
the work being on xwen itself. Nobody found a math defect. What the round changed:

- The oracle script defaulted to CPU, which against the Metal fixture sits right at the
  bars; it defaults to Metal now and the test refuses a fixture not made on Metal.
- The bars had 3x slack and pooled every position, so a window or ring bug confined to a
  few rows would have passed. Added: the regression bars, the worst-window bar, a run at
  the shipped 2048 chunk, the window-512 bracket, the exact ring test and the chunk test.
- The f32 attention was both the production path and the frozen oracle; it is two arms now.
- A wrong-graph switch was read from the environment at every load, so a stray value
  would have served a broken model; it is a load option only the test passes.
- The mask was rebuilt on the CPU per layer (about 285 MB per full layer at a 2048 chunk
  into 32k); it is built once per kind per forward.
- Admission priced Flint at a flat 8 GiB of scratch; it is the fit above.
- Codex found that serve's OpenAI normalizer folded `developer` into `system` and that
  its Qwen turn shaping (merging same-role turns, demoting a late system message to user)
  rewrote Flint conversations. There is a `Message::Developer` now, rendered
  `<|developer|>` on Flint and as a system turn, byte-identical, on every Qwen dialect,
  and on Flint `push_turn` renders turns in place. Both the OpenAI and Anthropic
  normalizers take the target's dialect.
- The shared-expert gate keyed on the routing kind; it keys on the tensor now.

## Not taken now

- **Inline `<think>` in replayed assistant content stays text.** A client that replays
  the model's raw output in `content` without `reasoning_content` gets its reasoning
  encoded as plain bytes, where the template and llama.cpp would encode `<think>` as id
  150006. Clients using `reasoning_content`, which serve emits, are unaffected. Reopen when
  a client that replays raw content is seen on Flint.
- **Recursive's Q4_1 "trained INT4" GGUF.** Unmeasured against the Q4_K_M, and xwen has no
  Q4_1 expert kernels. Reopen if the Q4_K_M's quality on agent work disappoints, priced by
  a perplexity A/B on llama.cpp first.
- **The developer turn cannot be checked against llama-server**, which rewrites the role
  before the template runs. Reopen if llama.cpp stops rewriting it, or grade it against
  transformers main once Flint is in a transformers release.
- **Serve-level snapshot, rewind and page-out on a Flint conversation** are not exercised
  end to end. The ring's own tests cover the cache. Reopen with the tool-call parser, which
  is the first serve work Flint will get.
- **The layer-46 expert-193 row.** More traced rows would turn the reading into a
  measurement. Reopen if the kernel arc's parity moves.

## Next

The kernel arc, Front of TODO.md: sinks in the prefill flash kernel and a decode kernel
with sinks, both graded against `FlintAttn::reference_attention`, then a sigmoid arm of
the fused router and an ungated epilogue, then a pinned bench for the first figures in
perf-state.md.
