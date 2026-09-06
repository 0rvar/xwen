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

**Dense Qwen3 arrives as BF16 safetensors, read one shard at a time and copied, not
aliased.** GGUF stays the only format for every other checkpoint; `qwen3` is read
through the pinned candle's own `MmapedSafetensors`, which adds no dependency and no
hand parser. Two deliberate departures from the obvious spelling. It opens one instance
per shard rather than `::multi`, because `multi` collapses a duplicated tensor name to
the last shard and `tensors()` loses shard provenance, which makes "present in exactly
its named shard" and "not in two shards" unimplementable; the loader owns its own
routing map instead, at the same mmap cost and on the same `load` path. And the payload
is copied into device buffers rather than aliased in place, because it cannot be
aliased: measured on all three sets, shards 1 and 2 start their data at
`8 + header_len` with `% 16 == 8`, and both `gguf::dense_alias_tensor` and
`ops::matmul_bf16` require a 16-byte start offset. That copy is what makes the alignment
irrelevant rather than fatal, and candle's `load` preserves the stored dtype, so BF16 in
the file is BF16 on the device with no silent widening. Norm planes are the one
exception, widened to F32 at load because candle's Metal `rms_norm` needs the weight
dtype to match the activations. Validation is CPU-only and happens before any
allocation: shard membership, duplicate names, duplicate index keys, stray
`*.safetensors` the index does not list, the expected shape table, BF16-only. `TensorSet`
is consume-once and `finish()` errors listing leftovers, so a plane nobody read is a
load failure rather than a silent zero. A `lm_head.weight` that ships anyway must be
byte-equal to the embedding (the config check refuses untied embeddings) and is then
struck off the ledger without being read, which is a 742 MiB device allocation not made.
The set's identity is a `gguf::CheckpointId` chained over `config.json`, the index and
each shard's header, so snapshots and the disk tier key a safetensors set exactly as they
key a GGUF; it hashes metadata only and sums whole-file lengths, which catches a shard
whose size changed and deliberately does not catch a same-length in-place overwrite
(2026-09-06).

**A projection plane with a long run of zeros is a load error unless the registry entry
allowlists it by name.** The scan is not defensive programming looking for a use: it was
written against a shipped defect. `Tongyi-MAI/Z-Image-Turbo`'s `text_encoder/` shard 3
carries `model.layers.35.mlp.up_proj.weight` with 14,772,816 contiguous zero elements
from element 27,003 and `down_proj` with 3,938,425 from element 20,930,265, where
`Qwen/Qwen3-4B`'s byte-identical-headered shard 3 has none - one contiguous run each,
not magnitude-selected and not row-structured, which is a torn write and not pruning
(docs/zimage.md). The threshold is a run strictly longer than 4096 elements, which no
trained plane in either set comes near. The allowlist is per registry entry rather than
per load, so the corruption is tolerated exactly when someone names the entry that
documents it and refused when a bare directory is pointed at; the same entry also
refuses the layer index that would evaluate those planes. The second scan in the same
pass counts values outside f16's range, because the tensor-core gemm stages BF16 weights
to half: 10,917 of 4,022,272,000 projection values below the subnormal floor on the base
set and 10,876 on Z-Image's, none above 65504, both reproduced independently. So the
gemv-versus-gemm difference on this checkpoint is a 1.1e-5 flush, not an overflow, and
that is a bounded fact to read parity numbers against rather than a hazard (2026-09-06).
