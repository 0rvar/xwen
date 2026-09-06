# The prefix cache and the disk tier

One topic of [docs/decisions.md](../decisions.md), the index of decisions by topic; moved
here verbatim on 2026-09-06. Dated paragraphs, newest additions appended within their topic.


Inherited from laguna; correctness now depends on snapshotting (KV cache for the 10–16
full-attention layers) + (conv + delta state for the linear layers) as one unit. Sizing
is favorable: the 35B keeps KV for only 10 layers with 2 KV heads — the hybrid's state
is far smaller per token than a uniform transformer's (2026-07-28).

**`CONTAINER_VERSION` stays at 2 for the DeltaNet snapshot variant — deliberately, not
by oversight.** The version discriminates FRAMING (header fields, directory layout,
record tags) and nothing else, because two mechanisms already cover the payload and
leave a bump nothing to catch. The checkpoint binding (hash plus file length, checked in
`read_header`) means an image can only be read back beside the exact file that wrote it,
so a laguna-era image cannot reach a Qwen build at all. Within a checkpoint,
`kv_cache`'s per-layer kind tags (`LAYER_FULL`/`LAYER_SWA`/`LAYER_LINEAR`) give each
kind its own field layout and dtype, and `check_restorable` rejects a layer whose kind
or shape disagrees with the live cache. The recurrent-state snapshot is therefore a new
per-layer tag inside unchanged framing. Bump the version only when the framing itself
changes; the invariant is recorded at the constant so the next reader does not have to
re-derive it (2026-07-28).
