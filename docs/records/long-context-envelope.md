# The 128k operational envelope

2026-09-06. The ledger item this closes had been open since 2026-08-11 and said the same
thing five ways: every perf figure in the repo was taken at `max_ctx 8192`, four
operational constants were sized against that world, and raising the default made long
contexts REACHABLE rather than characterized. This is the characterization, and the four
constants that follow from it.

Everything here was measured on the pinned worktree build of 4a66616 unless a section
says otherwise, `pmset -g` reporting `lowpowermode         0` — automatic, and high-power
mode is never claimed because neither key can confirm it. Another agent used the same GPU
throughout, taking `/tmp/xwen-gpu.lock` between its own runs; every run here waited for
that lock and held it for its own duration, and CPU load from that agent's builds reached
15.8 while some rows were taken. Read the shape rather than the absolutes.

## The instrument

`scripts/longctx.ts` is new and is the thing to reuse. It synthesizes a prompt of N
tokens by cutting repo prose (the bench fixtures plus everything under `docs/`) to a
byte length found by secant iteration on `llama-tokenize` against the checkpoint's own
GGUF vocab — a vocab-only load, ~0.4 s, so converging on a length costs far less than one
wrong prefill. Prompts are cached under `/tmp/longctx/` and are byte-identical across
sessions.

Two harness decisions are worth keeping.

**Decode has to be forced to be a rate.** The first version fed the prompt raw. At 8192
that decoded 64 tokens and looked fine; at 32768 it decoded ZERO, because a raw prompt
stops in the middle of whatever document the cut landed in and the model's most likely
next token there is an end-of-generation one. The fix is the chat template plus a
`--min-think` floor, which guarantees at least 160 decode tokens at every length and is a
realistic long-document turn besides. Any harness that reports a decode rate off a raw
continuation of a truncated corpus should be checked for this.

**Lengths are interleaved.** Running every repetition of one length before starting the
next makes thermal or contention drift look like a property of the length. The harness
runs all cells at pass 1, then all cells at pass 2. Above 65536 it also discards the
first repetition instead of setting `XWEN_BENCH`, because an in-process warm-up doubles a
131072 prefill and would have pushed those runs against the 20-minute box for no extra
information.

`scripts/bench.ts` grew the `--bin` flag it was documented as having. On `longctx.ts`
prefer the `XWEN_LONGCTX_BIN` environment form: the repo's `pgrep` guard matches any
command line containing `target/release/xwen`, so a `--bin` naming a worktree build makes
the harness itself look like a running model process to every other bench on the machine.

## The envelope

The tables are in [perf-state.md](../perf-state.md), which is their single source. The
three findings:

**Flash-Next decode is flat across context and the 35B's is not.** 47.1 tok/s at 8243
tokens and 46.9 at 65596, against the 35B's 96.4 falling to 36.8 at 131382 — a 62% loss
on the dense-attention checkpoint against nothing measurable on the sparse one until
past 64k. QSA is doing exactly what it is for. The practical consequence is that the
35B's headline 127.0 tok/s describes short conversations only, and a long-context 35B
figure has to be quoted from this table instead.

**Prefill is the wall an operator actually hits, and it falls on both.** The 35B goes
2326 to 668 tok/s and Flash-Next 925 to 231, so a maximal prefill is 197 s on the 35B and
567 s on the default checkpoint. That number is what `queue_timeout` is now derived from.

**Peak footprint quadruples on the 35B**, 12.0 GB at 8k to 50.5 GB at 128k, on weights of
20.4 GB and a KV cache of 2.6 GB at 131072. About 28 GB of the peak is neither. The
prefill mask accounts for it: candle's Metal buffer pool keys on exact size, and each
chunk asks for a fresh mask buffer of a different size — 2048 x 2048, then 2048 x 4096,
up to 2048 x 131072 — which summed over the 64 chunks of a 131072 prefill is 34.9 GB of
distinct f32 buffers the pool retains one of each. `ops::chunk_sync`'s own doc comment
already named this shape before anyone measured it.

## The four constants

**(a) The prefill mask now builds on the device, and the claim that justified it is
refuted.** The ledger called the host fill "the binding cost of long prefill": a scalar
double loop over `seq x (pos + seq)` f32, ~8.6e9 stores over a full 131072 prefill, plus
an upload of the same plane. `PrefillMask::causal_on_device` replaces it with two
`arange` vectors, a broadcast compare and a `where`, giving bit-identical values —
`the_device_built_causal_mask_equals_the_host_fill` compares both the additive f32 plane
and the f16 sdpa copy at every chunk shape the prefill walk produces, on the real Metal
device. `XWEN_HOST_MASK=1` restores the host path and is the control arm for the
Flash-Next replay check.

It buys nothing in time and a great deal in memory, which is the opposite of what the
ledger predicted. Both A/Bs ran the working-tree build in both arms, so only the mask
path differed.

| Checkpoint, 131072 tokens | Prefill tok/s | Decode tok/s | Peak footprint |
| --- | --- | --- | --- |
| 35B-A3B, host fill | 667.8 | 37.0 | 42 GB (69 GB on the discarded pass) |
| 35B-A3B, device build | 659.2 | 36.6 | 17 GB (17 GB on the discarded pass) |
| Flash-Next, host fill | 230.8 | 41.9 | 59 GB |
| Flash-Next, device build | 230.9 | 42.1 | 59 GB (53 GB on the discarded pass) |

On time it is a dead heat on both checkpoints, one repetition each. The likely reason is
that candle is asynchronous: the host fills chunk N+1's mask while the GPU is still
working on chunk N, so ~69 GB of host stores and uploads hide entirely behind a prefill
that is 197 s on the 35B and 569 s on Flash-Next. So the ledger's "binding cost of long
prefill" is refuted as a TIME claim, and nobody should quote this change as a throughput
win.

On memory it is the answer to the ~28 GB question above. The 35B's peak at 131072 goes
from 42-69 GB to a flat 17 GB — a 25 to 52 GB reduction, and the device arm's two passes
agree exactly where the host arm's differ by 27 GB. The mechanism is candle's Metal
buffer handling: the host path reaches the device through `Tensor::from_vec`, one fresh
exact-size buffer per chunk with no two chunks asking for the same size, while the device
path's `where` allocates through the pooled builder and gets recycled. The 35B at
`max_ctx 131072` now fits comfortably beside its 20.4 GB of weights instead of running
at three times that.

Flash-Next does not move, and the reason says what to do next: its QSA indexer builds
its OWN `n x n_kv` f32 mask on the host, per sparse layer per chunk, through the same
`Tensor::from_vec`, and above the 2048 budget that is every chunk. That is what the
remaining 42 GB is. The QSA mask was NOT moved in this arc: the criterion given was 10%
or more of the 131072 prefill wall, and the causal mask — the same kind of work through
the same call — priced at 0% of wall, so the criterion fails on its own terms. It is
ledgered instead with the number this arc gives it, which is a memory number and not a
time one.

**(d) `DEFAULT_DRAFT_CTX` stays at 8192, and the reason is stronger than "not
re-derived".** The criterion was the largest of 8192 / 16384 / 32768 at which the drafted
35B arm beats plain by 10% with the drafter cache under 2 GB. The memory half passes at
32768 (1.57 GB at 48 KiB/token). The throughput half fails everywhere:

| Prompt tokens | Plain tok/s | Drafted tok/s | Delta | Acceptance |
| --- | --- | --- | --- | --- |
| 1046 | 121.9 | 111.9 | -8% | 80.6% |
| 4117 | 116.3 | 85.3 | -27% | 70.9% |
| 8201 | 104.2 | 73.1 | -30% | 58.5% |
| 16409 | 99.1 | 62.8 | -37% | 57.4% |

At 1046 tokens the drafter accepts 80.6%, inside the band the shipped fits were made at,
and still loses 8%; the loss then deepens with context. So raising the horizon would
extend a loss, and the 8192 that was "inherited from laguna and never re-derived" turns
out to be limiting damage. That is the finding, and it is not the one anyone expected.

The reason the shipped +26-28% and this -8% can both be true is in perf-state.md's own
note: the 35B drafted figures are against the PRE-FOLD plain level of ~105 tok/s, and
plain has since gained the fold, the router gemv (+10.3%) and the fused shared expert
(+1.6%) to reach 127.

The text-type caveat this arc would otherwise have to carry is closed from outside it.
The same day's presence-penalty A/B, on a CODE prompt at 256 tokens with three
interleaved reps on its own pinned build, read drafted 121.1 against plain 126.5. Two
independent workloads, two independent sessions, same direction. What this arc adds to
that item is the shape: the loss is not a fixed offset, it deepens from -8% to -37% as
context grows, so a retune that only looks at short prompts will not see the half of it.

The horizon also stopped being silent. One line is emitted on the round where
speculation goes dark, verified end to end:

```
xwen: drafting stops past draft_ctx 8192 tokens (position 16409); decoding plain from here — raise --draft-ctx to speculate further
```

It is armed per crossing rather than per process, so a long generation past the horizon
says it once and a conversation rewound back under it says it again when it crosses
again.

**(b) and (c)** are in [decisions/serving.md](../decisions/serving.md): `queue_timeout`
derived from `context_length` at the measured prefill floor, and the disk flush budget
derived from the bytes actually queued with the old grace as its floor.

**(e)** is in [decisions/defaults-and-cli.md](../decisions/defaults-and-cli.md): one
`DEFAULT_MAX_CTX` behind the three clap defaults, and the `n_ctx_train` clamp moved to
`XwenModel::load` where every surface passes through it.

## Verification

All three gates were run on the working tree, which contains other agents' concurrent
edits to the sampler and metrics paths; none of those touch the prefill or mask paths.

`cargo test --release`, whole suite, Metal device present:

```
test result: ok. 1141 passed; 0 failed; 34 ignored   (lib)
test result: ok. 12 passed; 0 failed; 0 ignored      (binary)
test result: ok. 69 passed; 0 failed; 3 ignored      (parity harness)
```

`bun scripts/parity-gate.ts` on the 35B — the mask change touches the graph, so the gate
is not optional:

```
  PASS    strict (classic mv fallback) code-short  cos=1.000000 top5=5/5
  PASS    mm                           code-short  cos=0.999618 top5=5/5
  PASS    decode                       code-short  63/64 agree, 1 excused, 0 mismatch
  PASS    decode                       text-mixed  62/64 agree, 2 excused, 0 mismatch
  PASS    decode                       long-mixed  61/64 agree, 3 excused, 0 mismatch
  PASS    ppl                          corpus      Δnll=0.001179
ALL PASS (6 graded)
```

`bun scripts/flashnext-replay.ts --control XWEN_HOST_MASK=1`:

```
PASS: code-short 62/64 (2 excused, 0 hard); text-mixed 64/64 (0 excused, 0 hard); long-mixed 59/64 (5 excused, 0 hard)
```

185 of 192 agree, zero hard mismatches, inside the documented 185-189 band for this
checkpoint. Given that the two mask arms are bit-identical by test, that band is the
checkpoint's standing near-tie behaviour and not something this change moved.

## What was not done

Per the intake rule these are reasons and reopen conditions, not ledger items, except the
two that carry numbers and became items.

**The serve page-out write rate was not re-measured**, so `DISK_WRITE_FLOOR_BYTES_PER_SEC`
is 700 MB/s rounded down from the existing ~4.2 GiB / ~5 s figure in `DISK_FLUSH_GRACE`'s
own comment rather than from a run of this arc. It is the weakest number here. The shape
of the fix does not depend on it — a floor plus a rate is right precisely because the
small-image line `113 MiB in 1229 ms` is fixed cost rather than rate — but the rate
should be replaced the next time anyone runs `serve --no-tui` with the disk tier on and
watches a `paged out ... in N ms` line at a real conversation length. Reopen: with that
line, from a 32k-or-longer conversation.

**No profiled run was taken at 131072.** `XWEN_STACK_PROFILE` would have ranked the
stages, and the plan was to use it to price the QSA mask. It became unnecessary: the
causal mask's own A/B priced the same class of work at 0% of wall directly, which is a
better instrument than a profiler that is documented as reading 2.2x high on prefill.
Reopen: if anyone wants the QSA mask's TIME cost specifically, which this arc says is
not the reason to move it.

**The serve conversation benchmark was not run.** Decode at parity with `generate`
through 32k, the ~0.5 s resume and the re-prefill-on-edit are all from 2026-08-30 and
nothing in this arc touched the paths they exercise. Reopen: with the page-out rate
above, in the same sitting.

**Two items were opened with numbers.** The QSA indexer's host mask is 42 GB of
Flash-Next's 59 GB peak at 131072 and is now a measured item in "Cache images, memory and
context". The drafting-below-plain finding was NOT opened as a new item — the same day's
presence-penalty arc had already opened one from a code prompt, so this arc's context
curve was added to that item instead of duplicating it.

**One thing this arc did not touch and probably should be looked at.** AGENTS.md's
"Drafting" section still describes speculation as a both-checkpoint win at +26-28% on the
35B. Two independent measurements now read it below plain. Correcting that text belongs
with the retune that settles what the 35B should default to, not with this arc, which
changed no drafting default.


Listed here rather than as new ledger items, per the intake rule, except where a number
makes an item.

## Reviews and follow-ups (2026-09-06)

The three-model review of the day's commits (see the presence-penalty record) landed in
df5e678: the shutdown watchdog's grace now derives from the flush budget it was cutting
off, in-flight writes count toward pending bytes, and the horizon line fires at the
dispatch gates too. The QSA host-mask item this arc wrote as memory-only was folded into
a Front-1 prefill item the same evening (8d0c2fe): the causal-mask dead heat does not
transfer to a mask that waits on a per-layer readback, and the 925 → 231 tok/s prefill
curve is that path's number until a timer says otherwise. Drafting is off by default on
the 35B-A3B from the same commit, on the horizon curve measured here.
