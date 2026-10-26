# Flint

One topic of [docs/decisions.md](../decisions.md), the index of decisions by topic: the
Flint checkpoint (`ggml-org/Flint-GGUF`, arch `flint`) and everything decided about it.
Dated paragraphs, newest additions appended. The architecture and its traps are in
AGENTS.md's Flint section, the arc is
[docs/records/flint-correctness.md](../records/flint-correctness.md), and the parity
track is [docs/parity.md](../parity.md) "The Flint track".

**Flint is a design target, admitted on 2026-10-25 by the owner.** The design-target line
in AGENTS.md named the Qwen 3.6 and 3.8 GGUF checkpoints and Flash-Next; Flint joins them
as a tok/s target, which is a target amendment and not the scope amendment the dense 4B
and the image pipelines got. It is a 39.8B-total, ~2.5B-active MoE trained from scratch by
Recursive, Apache-2.0, built for local coding agents, and its active bytes per token put
its decode ceiling in the 35B-A3B's range (an estimate from the tensor table, not a
measurement). It shares no weights, vocabulary or template with any Qwen checkpoint here,
so it is a new architecture assembled mostly from parts the repo already had. The arc
that admitted it is correctness first: the attention is an f32 chain and speed is the
next arc (2026-10-25).

**The checkpoint is ggml-org's Q4_K_M GGUF, not the BF16 safetensors and not Recursive's
Q4_1.** The GGUF path reuses the loader, the quantized projections and the MoE kernels the
35B-A3B already runs, and llama.cpp runs the identical file as the oracle. The BF16 set is
80 GB and would need a second MoE loader for no target the repo has. Recursive's own
`flint-int4-q4_1.gguf` (26 GB, "trained INT4 codes") is unmeasured against the Q4_K_M and
xwen has no Q4_1 expert kernels; it is a record line, not a choice made against it
(2026-10-25).

**The Flint oracle is a second pinned llama.cpp checkout, `reference/llama.cpp-flint` at
a37cc461, and the main pin does not move.** The main pin predates Flint support
(ggml-org/llama.cpp#30928), and moving it would move every calibrated floor of the other
checkpoints (ground-truth-and-parity.md "The oracle is PINNED to a commit, not tracked").
`scripts/build-llamacpp.sh` takes the checkout as an optional argument. This llama.cpp
revision only builds the server and cli targets with `LLAMA_UI_GZIP=OFF`, which the script
now passes for both checkouts (b767234, 2026-10-25).

**Flint's attention is a `Mixer` variant inside the shared layer loop, not a side stack
like `src/qwen3`.** Only the attention is new. The residual, the norms, the dense FFN, the
MoE block, the taps, the profiler and the lm head are the trunk's, on the same GGUF
quantized planes, so a separate stack would have copied the loop and the tap plumbing to
change one block. `Mixer::Flint(FlintAttn)` holds full and sliding layers alike, each
with its own optional rope; the FFN became a per-layer choice for the two dense lead
layers. The cost is that every `XwenConfig` literal carries a `flint: None` field
(2026-10-25).

**The attention has two arms, and `reference` is frozen.** `XWEN_FLINT_ATTN` selects
`production` (the default, `AttnImpl::SHIPPED`) or `reference`. Both are the same f32
math today. The reference is the oracle the kernel arc will be graded against and is never
optimized; the production function is the one a kernel replaces. Production already caps
its score tile at 256 MiB, which leaves the parity numbers unchanged and took the peak
footprint at 57k tokens from 41.3 to 31.9 GB (2026-10-25). Amended 2026-10-26: the
production arm is the kernels now, `flash.metal` with a sink at prefill and `attn_decode`
at one row, and the tile cap went with the chain. The reference arm is unchanged.

**The routing is its own kind, sigmoid then top-6 then renormalize, and the fused glue is
off for it.** `Gating::Sigmoid` runs the candle chain. The fused router kernel computes a
softmax and the fused epilogue applies the shared-expert sigmoid, so both are gated on
softmax routing AND a gated shared expert, two separate conditions. Whether a shared
expert is gated is read from the file (`ffn_gate_inp_shexp` present or not), not inferred
from the routing kind. The sigmoid arm of the fused router is kernel-arc work
(2026-10-25). Amended 2026-10-26: the fused glue is on for Flint now, through a sigmoid
branch in the router and ungated epilogue kernels ("The fused MoE glue takes sigmoid
routing" below).

**The parity bars admit anything at least as close to llama.cpp Metal as llama.cpp CPU
is, and two wrong graphs fall far outside them.** llama.cpp's own CPU and Metal backends
disagree on this model by 0.613 mean |Δlogit| over the oracle's top-10, 0.863 top-5
overlap and 0.907 argmax agreement: sigmoid routers sitting on near ties flip experts on
last-place rounding, and the flip propagates. A bar tighter than that would grade the
oracle's backend. The acceptance bar is argmax 0.88, top-5 0.85, mean 0.65; xwen reads
0.970 / 0.949 / 0.210 at prefill. Dropping the sinks reads 0.047 / 0.080 / 11.98 and rope
on every layer 0.642 / 0.555 / 2.84, so the bar separates the model from a broken one. A
regression bar sits just outside today's figures (0.955 / 0.93 / 0.26 at prefill) and a
worst-64-position-window bar catches a fault confined to a few rows. A window one short
(512) reads 0.969 / 0.945 / 0.214, inside the oracle's noise, so no logits bar can see
it; a unit test holds the ring, the mask and both attention arms to the window's
definition at 1e-5 instead, and asserts it sees 512 at 5.9e-4 (2026-10-25).

**Chunk invariance is graded up to the first router and not past it.** Past layer 2's
router the logits are not chunk-invariant on this model even with every arm on the f32
reference path: a 300-token chunk against one 700-token chunk first moves a logit by 0.56
at position 45. A near-tie expert flips and attention spreads the changed position. So
`tests/flint_chunks.rs` grades hidden states at layers 0 to 2 across every chunk shape a
forward can see, which is where a ring, rope-offset or mask bug would show (2026-10-25).

**Thinking off is a no-op on Flint, not an error and not an empty think block.** The
template has no switch and always lets the model decide whether to open `<think>`, and
SGLang's reasoning parser ignores the flag the same way. Rendering `<think></think>` after
the header, the way GLM-style templates spell "off", is a prompt Flint's own template never
writes. So `--no-think` and `enable_thinking: false` render the same bytes as thinking on.
This differs from Qwen3-4B-Instruct-2507, where the flag is a 400; there the template has
no reasoning mode at all, here it has one the client cannot turn off (2026-10-25).

**Tools render but serve refuses them until a Flint call parser exists.** The CLI renders
Flint's tools preamble and its `<arg_key>`/`<arg_value>` calls byte-exact against
llama-server. Serve's parser reads the Qwen `<function=` form, so a Flint reply's calls
would come back as raw text; a tools request on a Flint target is a 400 instead
(2026-10-25).

**The Flint tokenizer is embedded in the binary, like the Qwen 3.6 one.** The GGUF path
has no tokenizer of its own (`from_gguf` is a stub) and the GGUF carries only the vocabulary
and merges, so `reference/flint/tokenizer.json` is vendored and `include_bytes!`'d, and a
test holds it token-for-token equal to the GGUF's `tokenizer.ggml.tokens`. Lookup for the
family is the embedded copy with no file search. The stops are the four ids the vocabulary
resolves by text (150001, 150003, 150005, 150018); the GGUF advertises only 150001, and
the Qwen stop ids are never folded in. A server builds Flint's vocabulary and grammar trie
only when a Flint checkpoint is selectable (2026-10-25).

**Flint's decode attention is a vendored kernel, GQA-grouped, not candle's `sdpa_vector`.**
candle's one-row kernels take no sink, assume K/V rows packed at the head dim so a cache
capacity view or the ring would need a copy, and run one threadgroup per query head, so
at Flint's GQA of 8 every KV row is read 8 times. `attn_decode.metal` is derived from them
and differs in all three: one threadgroup per (split, KV head) holds the group's 8 query
heads and reads each K/V row once through explicit head strides, and the sink is folded
into the merge. Keys are split at a floor of 32 per split, capped at 128 threadgroups in
all, which the microbench fitted as 16 splits at the 513-key ring and 32 at 4k and 32k. At
32k keys it reads 188-250 µs a call against 3.7-5.6 ms for the f32 chain, ~1.5x the read
floor (low power, unpinned). A simdgroup-MMA variant that could close the rest is a
record line (records/flint-kernels.md "Not taken now") (3063e0e, 18e1294, 2026-10-26).

**Flint's prefill attention is `flash.metal` with a `has_sinks` flag, and callers without
sinks are bit-identical.** The kernel already computed the causal edge and the window from
`pos`, `k_off` and `window`, so the sliding layers needed no mask, only the sink. With the
flag set, a row's running max starts at `sink * log2(e)` and its sum at 1; without it the
seeds are `-inf` and 0 as before, chosen at runtime, so the Qwen3 dense and Z-Image
callers run the same arithmetic and every existing flash test is still bitwise. A second
kernel would have meant a second copy of the block skip and the window rule. Sinks at
-1e30 are bitwise against the sinkless kernel, which is what keeps the `NoSinks` bracket
meaningful on the production arm (3063e0e, 2026-10-26).

**The production attention takes geometry, not a mask.** `flint::attention::visibility`
turns the cache's kind and the position into `(k_off, window)`, the kernels apply it, and
`forward` refuses a position that disagrees with the cache length. Masks are built only
when the reference arm is selected. This removes the per-forward host mask (268 MB at a
2048 chunk into 32k). The cost is that a window off by one now lives in a helper and not
in `attn_mask_for`, and no logits bar can see it, so the ring test calls the same helper
the forward does (18e1294, 2026-10-26).

**The fused MoE glue takes sigmoid routing and an ungated shared expert; the router is
bitwise, the ungated shared-expert pair is held to the gated pair's tolerance.** The
router's sigmoid branch replaces only the softmax step and is bitwise against
`route_sigmoid_from_logits`, saturated ties included, because a sigmoid router on near
ties is exactly where a last-place difference flips an expert. The ungated epilogue is
bitwise against `ops::combine` plus the add. The fused shared-expert pair cannot be: the
gated pair reassociates its dot products and has always been graded by tolerance, and the
ungated variant runs the same templated body, so it gets the same bars (rel L2 1e-5 against
an f32 host oracle) plus bitwise identities against the gated kernels at a gate of exactly
1.0. Gatedness is still read from the file. A Flint MoE layer goes from 19 dispatches to 8
at decode (3063e0e, 2026-10-26).

**Flint's memory admission is a flat scratch plus a term in the KV capacity, refitted to
the kernel arm.** With no score tile and no f32 copy of the cache the peak beyond the
weights is 7.3-7.9 GB to ~60k tokens and then follows the KV capacity, 16.3 GB at 198k.
`flint_scratch_bytes` is `(7 GiB + 48 KiB per KV slot) x 1.05`, 11-40% above six measured
points, priced at the KV capacity. The ~60k refusal is gone and `--max-ctx` is the limit.
The reference arm is not in the fit; it is a bisect arm (23fcd68, 2026-10-26).
