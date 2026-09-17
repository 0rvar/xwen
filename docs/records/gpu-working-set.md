# A deep prefill runs out of GPU working set

2026-09-17. Two client sessions resuming long cached conversations failed every time,
both at the drain that closes a prefill span, both with zero output tokens:

```
Metal error Command buffer had following error:
Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)
```

One was resuming 85,126 cached tokens and prefilling 6,772 new ones, the other 47,397 and
4,016. A 12k-token conversation on the same server worked. `memory.jsonl` recorded the
process footprint going 25 to 43 GB during each attempt, and system use 120 to 133 GB of
137.

Nothing was leaking. The prefill asked the GPU for more than was left, the error surfaced
at `engine.device.synchronize()` rather than at the allocation, and the three levers below
bound what it asks for.

## The accounting

`recommendedMaxWorkingSetSize` on this device is 115,448,725,504 bytes, 107.5 GiB. What
holds a share of it for as long as Flash-Next is resident:

| | bytes | note |
| --- | --- | --- |
| weights | 93.2 GB | UD-Q4_K_XL, four shards |
| KV cache | 3.22 GB | 24,576 B per position at 131,072 slots |
| QSA indexer planes | 2.01 GB | 7,680 B per position x 12 layers, allocated at `max_ctx` up front |
| left for a forward | ~16.9 GB | |

A prefill forward's transients are the mask planes, the score tile, and on the sparse-tile
route the gathered key and value columns. All three are proportional to the chunk times
the cache length, and candle's pool rounds each allocation up to a power of two and holds
it until a wait prunes it. At the shipped 2048-token chunk 92k tokens in, that measures
about 18 GB. Against 16.9 GB of room it does not fit, and it fits less every turn the
conversation grows.

The 131k prefill in [perf-state.md](../perf-state.md) peaked at 31.0 GB of footprint and
ran fine on 2026-09-06. It ran beside 77 GB of weights. The current default file wires
93.2, which is 16 GB of the headroom that measurement had.

## Three levers

**The chunk tapers with the context.** `Arch::prefill_chunk_at(pos)` holds the fitted
width up to `ops::QSA_SPARSE_MIN_KV_DEFAULT` (49,152, the sparse gate) and halves it for
each doubling past that, floored at 512:

| cache position | Flash-Next / 35B-A3B | dense 27B / Qwen3-4B |
| --- | --- | --- |
| 0 to 49,152 | 2048 | 512 |
| 49,153 to 98,304 | 1024 | 512 |
| above 98,304 | 512 | 512 |

That keeps chunk times cache length flat instead of letting it grow linearly with the
conversation, and it costs the fitted prefill rate only past the point where the sparse
route has already taken over from dense attention. Serve's span loop and
`Generator::prefill_tokens` both ask for the width at each chunk's own start position, so
a span that crosses a boundary narrows inside itself. The width is monotonically
non-increasing in position, which is what makes re-asking safe: a chunk never widens
mid-prompt. `XWEN_PREFILL_CHUNK` and the new `--prefill-chunk` pin one width at every
position instead.

**The dead mask is not built.** `run_stack_hc` hoisted a causal mask for every forward of
more than one token: f32 `[seq, pos + seq]`, a u8 predicate and an f16 sdpa copy, 1.75 GiB
together at a 2048-token chunk 92k in, held across all 48 layers. `AttnBlock::forward`
reads it only when the indexer returns `Dense`, which happens only while the whole cache
fits the 2048-token indexer budget. Above the budget every QSA layer brings its own
per-query mask or the tile route's column union, and the hoisted planes were allocated and
read by nobody. `causal_mask_has_reader` decides it now: build the mask when some
full-attention layer either has no indexer at all or is still inside its budget. The
change is inert by construction, the mask being discarded in exactly the cases it is now
not built for, and the existing `force_dense_qsa` equivalence tests pin the arm that still
reads it.

**Chunks stop pipelining past the gate.** Above 49,152 the span loop drains the device
between chunks, so the pool holds one chunk's garbage instead of two. Below it the chunks
pipeline as before. This is not a claim that syncing is faster: `decisions.md`
"Chunk-boundary device syncs and command-buffer batching granularity are both REFUTED as
levers on the 27B prefill residual" priced the sync at +9.2 µs/token at 925 tokens and
+2.4 at 4k, a fixed price per chunk, and found nothing length-dependent behind it. The
price is paid for residency here and the refutation stands for speed.

## The line in the log

Serve samples `currentAllocatedSize` and `recommendedMaxWorkingSetSize` before each
prefill span and estimates the span's transients at 96 bytes per (chunk token x cache
token), which is the 18 GB measured at 2048 x 92k divided out
(`memory::TRANSIENT_BYTES_PER_CHUNK_CONTEXT_TOKEN`). Over the free working set it says so
once per request, naming the allocated and recommended sizes, the estimate and the chunk.
It does not refuse: the estimate is an estimate, the tapering chunk is the mitigation, and
a refusal would cost a conversation that would have finished. The point is that
`~/.local/state/xwen/serve.log` carries the sizes when the drain returns an
out-of-memory that names none of its own.

The same constant sizes the admission estimate. `language_peak_bytes` carried a flat 8 GiB
of scratch, which covered dequantized planes and short-context prefills and was written
before context made the transients the larger term; it is now the widest forward the load
can reach, the tiered chunk at `max_ctx` over a full cache, floored at the old 8 GiB so a
small window admits exactly as it did.

## Verification owed

The mask guard needs none. It is provably inert: the planes it stops building were
discarded unread in exactly those cases, and the tests covering the case where they are
read are unchanged.

The chunk tiering changes which forwards run, so greedy output at a fixed seed has to be
checked equal across the tier boundary, and it has not been.
`bun scripts/flashnext-replay.ts` is the check ([parity.md](../parity.md)), and it could
not run in this session: the live `xwen serve` held Flash-Next, and one large model
process at a time is the rule on this machine. Owed at the next window, with the fixture
prefilled at a position past 49,152 so the boundary is actually crossed. The arithmetic
per forward is unchanged by construction, the chunk being a batching decision, so what the
replay would catch is a chunk-width dependence nobody has claimed exists.

Nothing here was benched. The three levers are memory bounds; whether the narrower chunk
costs prefill rate above 49k is unmeasured, and the 2026-08-30 fit that chose 2048 was run
at 3.8k tokens, where none of this applies.

## Not taken now

- **Growing the indexer planes with the cache.** 2.01 GB is allocated at `max_ctx`
  whatever the conversation holds, `IndexerCache` having no growth path. At the default
  262,144 window that is most of a chunk's worth of working set held for positions no
  conversation reached. Reopen if the headroom line fires on a server whose conversations
  stay short.
- **Refusing a prefill that does not fit.** The estimate would have to be trustworthy
  first, and one calibration point is not that. Reopen when the line has fired often
  enough to compare against what actually failed.
- **A chunk fitted to the headroom rather than tiered by position.** The device knows its
  free working set and could solve for the widest chunk that fits it, instead of halving
  on a schedule. That is the better shape and it needs the estimate to be trustworthy,
  same condition as above.
