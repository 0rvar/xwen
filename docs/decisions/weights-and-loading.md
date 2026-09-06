# Weights and loading

One topic of [docs/decisions.md](../decisions.md), the index of decisions by topic; moved
here verbatim on 2026-09-06. Dated paragraphs, newest additions appended within their topic.


**GGUF is the only weight format**, loaded by laguna's mmap/no-copy path
(`newBufferWithBytesNoCopy` over the page cache, batch-registered residency, classic
full-copy fallback under `XWEN_LOAD_CLASSIC`). Nothing in that loader is
architecture-specific; only the tensor-name table changes (2026-07-28).

**Loader name traps, recorded here because each one silently produces a working-looking
model:** there is no `ffn_norm` — `blk.N.post_attention_norm.weight` is the PRE-MLP norm
(HF semantics), not a Gemma-style post-norm. There is no `ssm_in` — the DeltaNet QKV
projection ships as `blk.N.attn_qkv.weight` and the z-gate as `blk.N.attn_gate.weight`.
`blk.N.ssm_a` (no `.weight` suffix) already holds `-exp(A_log)`. `blk.N.ssm_dt.bias`
uses a `.bias` suffix. Full-attention `attn_q` is double-width with per-head interleaved
`[q_h, gate_h]` layout (2026-07-28).

**beta and alpha ship as two separate tensors; there is no fused `ssm_ba` on these
architectures.** The loader briefly carried a fallback that split a
`[2·v_heads, hidden]` `ssm_ba` on the theory that either conversion might exist. The
shipped headers settle it: both files have `blk.N.ssm_beta.weight` and
`blk.N.ssm_alpha.weight`, `[hidden, v_heads]` Q8_0 each, mapped one-to-one from HF's
`in_proj_b` / `in_proj_a`. `LLM_TENSOR_SSM_BETA_ALPHA` → `blk.%d.ssm_ba` does exist in
llama.cpp's arch table, but is referenced only by the `qwen3next` arch, which maps HF's
fused `in_proj_ba`. The fallback was removed rather than kept as insurance: a branch
that can never fire is a claim about the world that nothing will ever check
(2026-07-28).

**Persistent state is partition-dependent in its low bits, and that is accepted, not
denied.** The dual-storage attention planes (f16 GEMM above `Q8_DECODE_MAX_SEQ`=8,
raw-q8 GEMV at or below it) are deliberately not bit-identical per weight element, and
the same `Proj` feeds cache-mutating paths — KV writes, the DeltaNet conv window and
recurrence. So the same tokens partitioned differently into forward calls (prefix-cache
snapshot stops, verify batches of 9+ vs one-token decode) produce state differing in
low bits. A second-model review caught model.rs's rollback docstring promising bitwise
identity with a differently-partitioned counterfactual — an observed identity promoted
to a claimed one, the exact failure mode this file's preamble warns about. The
docstring now states the real guarantee: restores are bit-exact replays of recorded
bytes; cross-partition agreement is numeric and parity-gated; `XWEN_ATTN_DEQUANT` pins
one canonical representation when bitwise partition-independence matters. The
alternative — one representation always — roughly doubles the attention-projection
bytes streamed per decoded token (halving those bytes is why dual storage exists;
laguna measured the win when it shipped it), a real cost not paid for a low-bit
property nothing currently depends on (2026-07-28).
