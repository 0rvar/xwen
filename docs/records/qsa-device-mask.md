# The QSA prefill round trip, priced and removed

2026-09-06. The Front-1 ledger item said the Flash-Next long-context prefill tax was the
sparse selection's host round trip and asked for an hour's timer before a fix. The timer
confirmed it, the fix followed the same session, and this is the arc: the pricing, the
device path, the A/B, and what the 128k prefill still costs afterwards.

Every run here is `scripts/longctx.ts` on a pinned copy of the binary under `/tmp`, the
131072-token Flash-Next prompt the envelope record synthesized (131323 tokens by the
model's vocab, 131424 prefilled with the template), `--no-draft`, one kept repetition
after a discarded one, `pmset -g` reporting `lowpowermode         0`. High-power mode is
never claimed. The GPU was otherwise free.

## The pricing

`XWEN_QSA_TIMER` wraps the host arm of `QsaIndexer::select_with` for a prefill chunk: an
explicit device drain first, so the readback is timed as a copy and not as the compute it
would otherwise wait for, then the score readback, the host top-k, the mask fill and the
upload on their own clocks. Everything after the drain until the upload returns is time
the GPU has nothing queued — that sum is the cost, and the drain itself is reported
separately as "sync-wait" so nobody adds it in. Totals print after the prefill
(`XwenModel::dump_stack_profile`).

Two repetitions on the host arm (the shipped path until this arc), 768 round trips each
(12 QSA layers x 64 chunks):

| | rep 1 | rep 2 |
| --- | --- | --- |
| prefill wall | 564.1 s (233.0 tok/s) | 563.1 s (233.4 tok/s) |
| GPU-idle, total | **102.4 s** | **106.5 s** |
| readback | 20.9 s | 21.7 s |
| host top-k | 41.5 s | 43.5 s |
| mask fill | 9.8 s | 10.1 s |
| upload | 30.2 s | 31.2 s |
| sync-wait (queued device work, not idle) | 335.4 s | 320.4 s |
| bytes | 105 GB of scores read back, 421 GB of masks uploaded | same |

18-19% of the prefill wall, and the host top-k — `select_nth_unstable_by` over 2048 rows
of up to 32856 blocks — is the largest piece, not the transfers. The largest chunk (the
352-query tail at n_kv 131424) cost 12 ms of readback for 46 MB, 16 ms of selection,
18-26 ms of fill and 12-14 ms of upload for a 185 MB plane. One observation the timer
did not set out to make: this arm peaked at 48 GB against the untimed host arm's 59 GB,
presumably because the explicit drain lets buffers return to the pool earlier. Not
pursued; the device arm below makes it moot.

## The device path

`kernel_qsa_select_mask` (`src/ops/qsa_select.metal`) is the decode selector's radix
select run one threadgroup per query over the `[n, n_blocks]` score plane, each
threadgroup writing its query's row of the `[n, n_kv]` additive f32 mask in place:
`-inf` across the row, a device-memory barrier, then zeros over the raw tail and the
selected blocks. The compaction scan of the decode kernel is not needed — a mask is
written by position — so only the equal-quota scan remains, which is the tie rule (lower
block index wins). Per row `nb = min((pos + i + 1) / ratio, n_blocks)` and
`keep = min(keep_max, nb)`; a row with `keep == 0` is tail-only and a row with
`keep >= nb` is the identity, both before any histogram so every barrier stays
threadgroup-uniform. `ops::qsa_select::select_mask` is the wrapper,
`dispatch::run_qsa_select_mask` validates (f32, contiguous, rank 2, `n_blocks * ratio`
within `pos + n`, i32 ranges, `pos + n` checked), and `QsaIndexer::select_with` takes it
for every Metal prefill chunk above budget. The mask buffer comes from candle's allocator
(power-of-two sizes, any free buffer at least as large reused) rather than the exact-size
buffers `Tensor::from_vec` makes, which the pool never hands out again.

Kill switch `XWEN_QSA_HOST_MASK=1` restores the host arm; `XWEN_QSA_CLASSIC` implies it.
`XWEN_QSA_TIMER` stays as the instrument and only ever times the host arm.

Held bit for bit by `device_mask_matches_host_mask_bitwise` (`src/qwen4exp/indexer.rs`):
tail-only rows, rows straddling the budget, the shipped 512-block keep at 30k, 65k and
131k, a final partial chunk, a full 2048-query chunk ending at 128k, the 1023/1024/1025
block width transition, all-equal and all-zero scores, and NaN/negative scores; plus
the three-arm scripted sequence `cached_block_keys_match_the_classic_recompute`, which
now compares the device mask against the host one at every prefill chunk it runs.
`bun scripts/flashnext-replay.ts --control XWEN_QSA_HOST_MASK=1` passes (62/64, 64/64,
59/64, 7 excused, 0 hard) — but its fixtures are 612 tokens, below the 2048 budget, so
the replay exercises the `Dense` short-circuit and not this kernel; the bitwise tests are
the evidence for the kernel, and the end-to-end check is the greedy A/B below.

## The A/B

| Flash-Next, 131424 tokens | Prefill tok/s | Prefill wall | Decode tok/s | Peak footprint |
| --- | --- | --- | --- | --- |
| host arm, envelope record (2026-09-06, earlier) | 230.8 | 569.3 s | 41.9 | 59 GB |
| host arm, timed (this arc, rep 1 / rep 2) | 233.0 / 233.4 | 564.1 / 563.1 s | 42.2 / 41.9 | 48 GB |
| device arm (rep 1 / rep 2) | 284.0 / 295.5 | 462.7 / 444.8 s | 42.1 / 41.4 | 28 GB |

**+23-28% at 128k, and the peak halves.** The 100-118 s the wall lost is the 102-106 s
the timer priced; the device path's own cost (writing 421 GB of `-inf` and the radix
passes over 105 GB of scores) does not show above the noise between repetitions. Decode
is unchanged, as it should be: a decode step already selected on the device. The device
arm's rows were taken on a build that rounded the mask buffer to 256 MB buckets; the
shipped binary drops that (candle rounds to powers of two itself, so it was redundant),
and the confirmation run of the shipped binary is the section below.

## Confirmation run of the shipped binary

Same harness, same prompt, the binary that ships (checked add, no bucket rounding):

| Flash-Next, 131424 tokens | Prefill tok/s | Prefill wall | Decode tok/s | Peak footprint |
| --- | --- | --- | --- | --- |
| shipped binary, rep 1 / rep 2 | 284.0 / 282.2 | 462.8 / 465.7 s | 40.9 / 41.3 | 28 GB |

Over the four device-arm repetitions of the day the 128k prefill spans 445-466 s
(282-296 tok/s) against 563-569 s on the host arm, so the claim is **+22-28% and a 28 GB
peak**, and the bucket rounding was worth nothing either way, as candle's own rounding
predicted.

**End to end, the arms decode the same tokens.** A greedy (`--top-k 1`) 8243-token
templated run with 192 decode tokens on the shipped binary, once with
`XWEN_QSA_HOST_MASK=1` and once without, produced byte-identical output — 978 bytes,
every token. That run is also the one figure below 128k: the 8k prefill read 947.8 tok/s
on the host arm and 1096.5 on the device arm, +15.7%, one repetition each with the
in-process warm-up, so the round trip was worth about a sixth of the wall there too.

## What the 128k prefill still costs, and not taken now

At 445-463 s the 128k prefill is still 3.2-3.5 ms/token against 1.08 at 8k, which is
still three times the 35B's growth. The round trip is gone, so the remaining growth is
the sparse layers' attention itself: `AttnBlock::forward` hands candle's sdpa the full
`[n, n_kv]` mask and sdpa computes every masked-out column and discards it — the
selection buys nothing at prefill except correctness, while a decode step gathers its
selected rows and attends over 2048 + tail. Not taken now: a prefill route that gathers
the union of selected blocks per query group, or a block-skipping sdpa, is a new attention
kernel with a rounding contract (the fused Metal sdpa is f16). Unpriced. The reopen
condition is a duplicate-dispatch probe (`XWEN_DUP_STAGE`, docs/benching.md) run at 128k
pricing `MixerFullAttn` on a QSA layer against the same stage on the 35B; if attention is
the majority of the residual growth, that kernel is the next long-context item. A smaller
thing also not taken: writing the mask as f16 in the kernel so `PrefillMask::from_raw`'s
f32-to-f16 pass over 421 GB disappears; same values, one fewer pass, unpriced and
probably a few seconds.

## Review

Codex reviewed the diff read-only and found one Low (an unchecked `pos + n` ahead of the
overflow checks — fixed with `checked_add` and a refusal test) and named three untested
shapes (the 1024-block width transition, a full 2048-query chunk at 128k, all-equal
scores), all three added to the sweep. It also caught the bucket-rounding comment
describing candle's pool as exact-size, which it is not for `new_buffer`; the rounding
was removed rather than the comment corrected. The Qwen review was skipped at the user's request, the GPU being the bench's.
