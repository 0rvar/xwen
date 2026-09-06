# 2026-08-08 — The 27B prefill residual is real and lives in the pipelining: per-stage syncs find only +103 of the +410-438 µs/token, and both cross-chunk accumulation and command-buffer batching are refuted as its mechanism

Moved verbatim from [docs/log.md](../log.md) on 2026-09-06; the log keeps this entry's
opening paragraph and links here.


**Context.** The dense-FFN gemm arc (P8c, 2026-07-29) closed the 27B prefill gap and
handed off one number it could not attribute: +350 to +560 µs/token of prefill cost
outside every measured stage, growing with prompt length, and named in TODO.md as the
largest remaining 27B-prefill unknown. That item's own next step was "per-layer timing
inside `model.rs` `run_stack` — in situ, not a synthetic bench". This arc built that
instrument and ran it. It shipped a diagnosis, not a fix.

**Built: an in-situ per-stage profiler (`XWEN_STACK_PROFILE`).** `src/stack_profile.rs`
decomposes each chunk's wall clock into the stages `run_stack` actually runs — embed,
mask fill and upload, per-layer attn_norm / mixer / residual / ffn_norm / ffn /
residual, final norm, lm_head — on the real weights in the real dispatch order. Four
design points carried the arc:

- **Stages are bracketed by device syncs, and adjacent stages share theirs.** A stage's
  total is completed GPU work, not enqueue time, and the brackets cost one sync per
  stage rather than two.
- **`inter_stage_host` is a bucket, not a leak.** The sync that closes a stage also
  opens the next one, so host-side gaps — loop bookkeeping, tap checks, tensor drops
  that free device buffers — would otherwise be charged to whichever kernel happens to
  follow them. They get their own bucket instead.
- **`unaccounted == 0` is an integrity invariant, not a finding.** Every interval inside
  a chunk's bracket belongs to some bucket by construction, so a nonzero residual there
  means the brackets are wrong. The residual being hunted has to show up AS a bucket.
- **Phase is declared by the caller (`XwenModel::set_phase`), never inferred from token
  count.** Two ways inference breaks: a prompt whose length is ≡ 1 mod 512 ends prefill
  with a one-token chunk, and a speculative verify forward feeds a whole span while
  being decode. `XWEN_BENCH`'s warm-up pass is excluded by resetting the accumulators
  after it, so a dump never mixes cold and warm chunks.

A second flag, `XWEN_CHUNK_SYNC`, makes the plain prefill loop wait for each chunk's
forward to complete before enqueueing the next. Both flags join `parity-gate.ts`'s
`baseEnv()` strip list. The diff got a two-model-family review (Claude + Codex
gpt-5.6-sol); the mask-construction refactor it required — splitting `kv_cache` into
`attn_mask_data` / `mask_tensor` / `PrefillMask::from_host` so the host fill could be
timed separately from the upload — was given a clean byte-identity bill by both.

**Round 1: the residual reproduces, and then most of it disappears under
serialization.** Conditions for every number in this entry: `lowpowermode 0` (this
machine exposes no `powermode` key, so high-power mode is never claimable), warm,
`XWEN_BENCH=1`, interleaved arms, medians of 3, 27B Q4_K_M, the committed prefill-925
(880 tokens) and prefill-4k (3851 tokens) fixtures. The plain-arm per-token length delta
between the two fixtures came out **+410.3 µs/token** in round 1 and **+437.9** in round
2 — the ledgered residual, reproduced twice.

Under per-stage serialization the same length delta is only **+102.8 µs/token**:

| stage | µs/token, 880 → 3851 |
|---|---|
| mixer_full_attn | **+53.5** |
| ffn | **+42.2** |
| residual_ffn | +16.8 |
| mask_upload | +7.9 |
| mixer_delta | −9.1 |

The attention row is the known quantity: ~+69 µs/token of sdpa+mask quadratic was
already the profiling pass's estimate, and +53.5 matches it. The FFN row is not — the
dense SwiGLU's per-token cost is length-INDEPENDENT by construction (the same weights,
the same work per token, no attention-like quadratic anywhere in it), so a stage that
grows +42 µs/token with prompt length is an anomaly in its own right. Its signature is
allocator pressure rather than arithmetic, and it is unexplained.

**So ~335 µs/token exists only when the stages pipeline.** Insert a device sync between
every stage and five sixths of the length-dependent residual stops being paid. That is
the arc's central finding and it reframes the item: the cost is not inside any stage's
kernels, it is in how consecutive stages' work interacts when the queue is allowed to
run ahead.

**Mechanism research against the pinned candle rev (21cca0b).** Four facts about
candle's Metal allocator and encoder that constrain the hypothesis space. Its buffer
pool prunes ONLY inside `wait_until_completed` / `flush_and_wait_current`, so nothing
reclaims within a free-running chunk. `Tensor::from_vec` bypasses the pool entirely,
allocating a fresh exact-size-keyed `MTLBuffer` plus a residency-set commit on every
call — the per-chunk mask upload is on that path. `find_available_buffer` scans O(total
cached buffers). And the encoder emits a full `MTLBarrierScope::Buffers` barrier
whenever a pool-recycled pointer is reused inside one encoder session. Default
`compute-per-buffer` is 50, overridable via `CANDLE_METAL_COMPUTE_PER_BUFFER`.

**Round 2: two candidate mechanisms, both refuted by direct A/B.**

*Cross-chunk accumulation is not it.* `XWEN_CHUNK_SYNC` does everything a chunk boundary
could do to reset accumulated state — it prunes the buffer pool, clears the fence map
and drops the encoder's barrier history. The length delta with it on is **+431.1
µs/token** against **+437.9** without: a difference of −6.8, i.e. nothing. (The flag
itself is not free — it costs +9.2 µs/token at 925 and +2.4 at 4k — which is a
consistent cost per chunk, not a length-dependent one.) Whatever accumulates,
re-accumulates inside a single 512-token chunk.

*Command-buffer batching is not it either.* `CANDLE_METAL_COMPUTE_PER_BUFFER` swept over
10 / 200 / 1000 against the default 50 at 4k: **all four means within 0.9%.** A 100x
range of batching granularity moves nothing, so the cost is not command-buffer
submission overhead and not a per-buffer fixed price.

**What survives, and where the instrument stops.** Two hypotheses are consistent with
every measurement and neither is confirmable with what is in the tree: intra-chunk
barrier storms from buffer-pointer recycling (the pool hands back a pointer already used
in this encoder session, the encoder inserts a full-buffer-scope barrier, and the
pipelining that the syncs removed is exactly what makes those barriers cost anything),
and fence-wait pileup (every new encoder waits on every fence in the `prev_ce_outputs`
map, which grows within a chunk). Both re-develop inside a chunk regardless of
chunk-boundary cleanup, which is precisely what the `XWEN_CHUNK_SYNC` result says must
be true of the real mechanism. Separating them needs a counter candle does not expose —
a patched candle or a Metal capture — and that is the boundary this arc stopped at
rather than guessing past.

**A thermal trap worth not repeating.** A warm-up pass that reads FASTER than the timed
pass following it looks like evidence about pool state, and is not: the warm-up runs on
the cooler chip and the ordering alone explains the gap. The profiler now excludes
warm-up from its dumps for this reason.

**Second finding: the "attention glue" ledger item's premise is inverted.** The item
proposed routing the main attention block through the existing bit-identical
`attn_gate` / `permute_01` / `cast_*` kernels, against a briefed ~42 ms/layer of unfused
eager glue. Reading the code says otherwise. `permute_01`, `permute_01_f16`, `cast_f16`,
`cast_f32` and `rope_neox` have been wired into the main attention block since the fork
(`attention.rs`'s `fused_glue` paths). `ops::attn_gate` has ZERO production call sites
and cannot serve Qwen as written — it computes a scalar-per-(token, head) softplus gate,
where Qwen needs a head_dim-wide sigmoid (attention.rs:360). DFlash uses no glue kernels
at all. So the ~42 ms/layer never existed, and the synced differential agrees: the
attention block's length-growth is +53.5 µs/token, ≈ the sdpa quadratic that was already
known. The item is downgraded, not closed — a fused sigmoid-gate kernel is still worth
~2-3 dispatches at the gate site, and the head-dim-256 flash instantiation still owns
the mask.

**Today's baselines, with the caveat attached.** The plain arm read **755-767 tok/s
@925** and **574 @4k**, against the ledger's 702 / 445 of 2026-07-29. Treat the deltas as
machine-state variance rather than as a regression or an improvement: the 27B's
between-run level shifts are documented at ±10% (TODO P11 caveat, decisions.md
"Measurement discipline"), nothing in this arc touched a production code path, and a
compile load preceded round 1. The residual differentials are what this arc measured;
the absolutes are context.

**Ledger.** TODO.md's "+350 to +560 µs/token" item is annotated with the split, both
refutations, the surviving hypotheses and the next step; the unexplained in-stage FFN
growth is called out there as a second thread. The attention-glue item is annotated with
the inversion. Both refutations are in decisions.md under "Refuted perf directions", and
the profiler's design and its differential-only reading rule are under "Measurement
discipline".
