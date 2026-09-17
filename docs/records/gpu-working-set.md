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
| weights | 93.2 GB | resident, from the incident's samples |
| KV cache | 3.22 GB | 24,576 B per position at the 131,072 slots it had grown to |
| QSA indexer planes | 2.01 GB | 7,680 B per position over all 12 QSA layers, allocated at the 262,144 `max_ctx` up front |
| left for a forward | ~17.0 GB | |

The two cache rows are quoted on different position counts on purpose: the KV cache grows
on demand and had grown that far, while the indexer planes are allocated at `max_ctx`
whatever the conversation holds. The weights row is not the file size. UD-Q4_K_XL is
111.33 GB in four shards, of which the 28.80 GB PLE n-gram table is mapped and read from
the host rather than uploaded, leaving an 82.53 GB trunk
([qwen4exp-port.md](../qwen4exp-port.md)); 93.2 GB is what the samples recorded resident
with the PLE working window and the dequantized planes on top of it.

A prefill forward's transients are the mask planes, the score tile, and on the sparse-tile
route the gathered key and value columns. All three are proportional to the chunk times
the cache length, and candle's pool rounds each allocation up to a power of two and holds
it until a wait prunes it. At the shipped 2048-token chunk 92k tokens in, that measures
about 18 GB. Against ~17 GB of room it does not fit, and it fits less every turn the
conversation grows.

The 131k prefill in [perf-state.md](../perf-state.md) peaked at 31.0 GB of footprint and
ran fine on 2026-09-06. It ran beside 77 GB of weights, so that measurement had 16 GB of
headroom a run today does not.

## Three levers

**The chunk tapers with the context.** `Arch::prefill_chunk_at(pos)` holds the fitted
width up to `ops::QSA_SPARSE_MIN_KV_DEFAULT` (49,152, the sparse gate) and halves it for
each doubling past that, floored at 512:

| cache position | Flash-Next / 35B-A3B | dense 27B / Qwen3-4B |
| --- | --- | --- |
| 0 to 49,152 | 2048 | 512 |
| 49,153 to 98,304 | 1024 | 512 |
| above 98,304 | 512 | 512 |

That holds chunk times cache length flat while there is width left to give up, and it
costs the fitted prefill rate only past the point where the sparse route has already taken
over from dense attention. Past the floor the product grows again at a quarter of the
slope: at the default 262,144 window a 512-token chunk at the end of the walk is 12.9 GB
against the ~17 GB above. This buys headroom, it does not bound the transients, and a
window much past 262k would need a lever this arc does not have.

Serve's span loop and `Generator::prefill_tokens` both ask for the width at each chunk's
own start position, so a span that crosses a boundary narrows inside itself. The width is
monotonically non-increasing in position, which is what makes re-asking safe: a chunk
never widens mid-prompt. `XWEN_PREFILL_CHUNK` and the new `--prefill-chunk` pin one width
at every position instead.

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
before context made the transients the larger term. It is now the widest forward the load
can reach, and that is not the width at `max_ctx`: widths step down while the cache grows
continuously, so the peak is the last wide chunk before a tier boundary. At a 65,536
window the 2048-wide chunk ending at 51,200 takes 10.1 GB where the 1024-wide chunk ending
at 65,536 takes 6.4, and `peak_prefill_transient_bytes` walks the window to find it. The
old 8 GiB is the floor, so a short window admits exactly as it did. At the default 262,144
window the term is 12.0 GiB, so a full-window language load now needs 4 GiB more to be
admitted than it did — which is worth knowing where the image engine's 40 GiB envelope
shares the coordinator.

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

- **The one-shot CLI paths keep the fixed chunk.** `Generator::generate`, the DFlash spec
  path and `prefill_mtp` still read `prefill_chunk()` at every position, and `xwen chat`
  re-prefills the whole conversation each turn, so a long CLI chat on Flash-Next
  reproduces the failing shape with only the mask guard between it and the same error.
  Left alone deliberately: the bench harness prefills through those loops, and the
  2026-09-06 figures in perf-state.md were all measured at a constant chunk, so tapering
  them silently would make the next A/B incomparable without saying so. Reopen the moment
  anyone hits the error outside serve, or when the >49k rows are retaken and the
  comparison no longer costs anything.
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
