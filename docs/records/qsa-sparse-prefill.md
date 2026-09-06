# The QSA layers attend sparsely at prefill

2026-09-06, the same evening as [the device mask](qsa-device-mask.md), whose record ended
with a probe as the reopen condition for the residual 128k growth. This is that probe,
what it found, and the route built on the finding.

Power: `pmset -g` printed `lowpowermode         2` for every run here; the owner had
switched the machine to high performance before the probe started. The line is quoted,
not read as a claim ([benching.md](../benching.md)). One reading off it: the Flash-Next
base arm read 465.9 s at 128k in this mode against 462-466 s on automatic an hour
earlier, so at 128k prefill the mode moves nothing, as it moved nothing at 3.8k on
2026-09-05.

## The probe: attention is half the 128k prefill, on both checkpoints

Three stages were added to the duplicate-dispatch probe for this: `sdpa` (the
multi-token attention kernel on every full-attention layer, both architectures),
`qsa_score` (the indexer's block scoring) and `qsa_mask` (the device mask plus its f16
copy). `/tmp/probe128k/probe.ts` ran `generate --raw -n 4 --stats` on the envelope's
131072-token prompts, arms interleaved and reversed for round 2, the GPU lock taken per
run, on a pinned build.

| 131311-token prefill | round 1 | round 2 | delta |
| --- | --- | --- | --- |
| Flash-Next base | 465.9 s | BASE_R2 | |
| Flash-Next sdpa duplicated | 706.6 s | out of memory | **+240.7 s, 52% of wall** |
| Flash-Next qsa_score + qsa_mask duplicated | 434.6 s | 449.2 s | below the spread |
| 35B-A3B base | 200.3 s | 203.7 s | |
| 35B-A3B sdpa duplicated | 361.8 s | 359.9 s | **+156-161 s, 77-81% of wall** |

The round-2 Flash-Next sdpa arm died at 64k positions with
`kIOGPUCommandBufferCallbackErrorOutOfMemory`: doubling the attention doubles its
transient buffers, and at 128k the second copy no longer fits beside the first. Round 1
stands as a single sample, eight times the run-to-run spread seen today (435-466 s on
the base shape), and the 35B's delta replicated. The indexer's own work (scoring,
selection, mask) reads at or below the noise floor — one of its two samples came in
under base.

So half of a 128k Flash-Next prefill is the 12 sparse layers running dense attention
over 131k columns per query, of which the mask then keeps 2048: about 20 s per layer,
the same class as the 35B's 16 s per dense layer. The selection was buying correctness
and nothing else at prefill.

## The union: how many columns a tile of queries really needs

`XWEN_QSA_UNION_STATS` reads each chunk's device mask back and counts, for tiles of T
consecutive queries, how many DISTINCT blocks the tile's queries select between them —
the columns a tile-batched attention has to gather and attend over. On the 65536-token
prompt (31 chunks x 12 layers), as a share of the blocks in the prefix, the 12-layer
mean and the worst layer's worst tile:

| prefix | blocks | T=16 | T=32 | T=64 | T=128 | T=256 |
| --- | --- | --- | --- | --- | --- | --- |
| 8192 | 2560 | 45.8 / 60.4 | 53.3 / 68.2 | 61.2 / 76.4 | 68.7 / 82.9 | 75.5 / 87.7 |
| 32768 | 8704 | 22.8 / 37.2 | 31.0 / 49.6 | 40.8 / 63.7 | 52.5 / 79.0 | 64.6 / 85.3 |
| 63488 | 16383 | 13.1 / 23.6 | 18.4 / 36.9 | 25.5 / 50.9 | 34.6 / 66.9 | 45.6 / 79.9 |

Neighbouring queries do not select the same blocks: each keeps 512, and 32 of them
together keep three to nine times that. Summing `n x S` over every layer-chunk of the
64k prefill, with S padded to each layer-chunk's largest tile as the route pads it, the
attention columns come to 28.0% of dense at T=16, 35.6% at T=32, 44.8% at T=64, 54.9% at
T=128. The sdpa's query tile is 32 rows, so T=16 would run half padding and lose its
edge; **T=32 ships** (`ops::QSA_TILE_DEFAULT`, `XWEN_QSA_TILE` to override). The share
falls with context, so at 128k it is lower than at 64k. Two thirds of the attention
is what this can remove at 64k, not the 98% a per-query gather would — a per-query gather
was rejected on memory traffic (2048 queries x 2051 rows x 2 KV heads x 512 B, 4 GB per
layer per chunk written and read back).

## The route

`ops::qsa_tiles` (`src/ops/qsa_tiles.rs`, `qsa_tiles.metal`). `kernel_qsa_select_mask`
now also writes each query's selected blocks as an ascending list beside the mask (the
tail's incomplete block last), pinned against the host sets by
`device_block_lists_match_host_top_blocks`. The indexer hands those over as
`QsaSelection::Sparse` — the same mask bits as `Mask`, plus the lists — and the attention
block, on Metal above the budget, runs:

1. `kernel_qsa_tile_union`: one threadgroup per tile, a bitmap over up to 65536 blocks
   in threadgroup memory, OR'd from the tile's lists, then compacted in word order so
   the union comes out ascending, with its count.
2. ONE readback: the tile counts (`n_tiles` u32), which size the padded column count
   S (the largest tile's union x ratio, at least T, rounded to 64). This is the sync
   the route keeps, and it waits on this layer's selection only.
3. `kernel_qsa_tile_gather_kv`: the union's K and V rows into `[n_tiles, 2, S, 256]`
   f16 from the cache's head-strided views; padding columns copy row 0.
4. `kernel_qsa_tile_mask`: the tile's `[T, S]` f16 additive mask read off the full mask
   at the union's columns, padding columns `-inf`, padding query rows copying the last
   real one; `kernel_qsa_tile_q`: the queries re-laid `[n_tiles, 24, T, 256]` f16.
5. candle's batched sdpa with the tiles as the batch, GQA as before; the output
   un-tiled by a permute-copy, narrowed to the real queries, widened to f32.

Every column a query needs is in its tile's union and every column it must not see is
`-inf`, so the result is dense masked attention's up to the online softmax's summation
order. `sparse_tiles_reproduce_the_dense_mask_route` holds it to the dense route at
4e-3 of the output magnitude on a real selection (2560-token prefix, a 100-query chunk
at a 256-token budget, the last tile padded); `tile_union_matches_the_host_union` and
`tile_gathers_match_host_gathers` hold the glue bit for bit. `XWEN_QSA_ATTN_CLASSIC=1` is
the dense route back, a parity row rather than a bitwise switch (parity.md).

## The A/B

`scripts/longctx.ts` on the pinned build, the same 131072-token prompt as every 128k row
today, `--no-draft`, one discarded and one kept repetition, `lowpowermode 2` (see the
top).

| Flash-Next, 131424 tokens | Prefill wall | Prefill tok/s | Decode tok/s | Peak footprint |
| --- | --- | --- | --- | --- |
| host selection (this morning) | 563-569 s | 231-233 | 41.9-42.2 | 59 GB |
| device mask, dense sdpa (this afternoon, 4 reps) | 445-466 s | 282-296 | 40.9-42.1 | 28 GB |
| **sparse tiles, T=32 (rep 1 / rep 2)** | **292.1 / 305.5 s** | **449.9 / 430.3** | 48.4 / 44.4 | 31 GB |
| sparse tiles, shipped gate at 49152 (rep 1 / rep 2) | 288.1 / 306.7 s | 456.2 / 428.5 | 48.0 / 48.6 | 31 GB |

**+45-62% over the device-mask build, and 1.9x over the morning's 231 tok/s**, for 3 GB
of peak. The gated row is the binary that ships (chunks below 49152 positions on the
dense route, above it sparse) and reads the same as the ungated one, four repetitions
spanning 288-307 s. The 240 s the probe put on the attention became ~90-100 s: the union statistics
predicted the attention columns at about a third of dense at this length and the wall
moved by about two thirds of the attention, which is the prediction. Decode is unchanged
(a decode step gathers rows and never enters this route).

**It is not a win at every length.** A greedy 8243-token run on the same binary read
1030.6 tok/s on the sparse route against 1096.5 on the dense one, -6%: below ~16k the
attention is cheap and the route's extra dispatches and its one readback per layer-chunk
cost more than they save. The crossover:

| Flash-Next prefill, same binary | dense route | sparse route | |
| --- | --- | --- | --- |
| 8243 tokens (greedy runs) | 1096.5 tok/s | 1030.6 | -6% |
| 16451 | 925 (17.8 s) | 828 (19.9 s) | -10% |
| 32921 | 754 (43.6 s) | 735 (44.8 s) | -2.5% |
| 65596 (2 reps) | 506-527 (124.5-129.6 s) | 594 (110.4-110.5 s) | +13-17% |
| 131424 (2 reps) | 282-296 (445-466 s) | 430-450 (292-305 s) | +47-59% |

One repetition per cell except where marked. So the route is gated on the cache length:
`XWEN_QSA_SPARSE_MIN_KV`, default 49152 (`ops::QSA_SPARSE_MIN_KV_DEFAULT`), the
midpoint of the 32k loss and the 64k win; below it an above-budget chunk keeps the dense
route over the mask, which is also the bitwise one. The exact break-even between 32k and
64k is interpolated, not measured. A side effect worth knowing: the dense 64k row here
reads 506-527 tok/s where the morning's envelope had 403, which is the device mask's
gain at that length, unmeasured until now.

**Numerically:** `sparse_tiles_reproduce_the_dense_mask_route` (4e-3 of magnitude on a
real selection), and a forced greedy replay of 64 steps after a 4096-token prompt with
the dense route as the reference and the sparse route as the candidate — 64/64 agree,
no near-tie needed (`logits-dump --greedy` / `--replay`, graded at the 0.5 band the
Flash-Next replay uses). The same replay at 8191 tokens could not run: the tool prefills
its prompt in ONE forward, and at n = 8191 the sparse route's per-tile operands (256
tiles, S near the whole prefix at that depth) pushed the process past Metal's memory
beside the 77 GB of resident weights; the dense route at the same length ran. The
production prefill chunks at 2048, where the route peaked at 31 GB over a 131072
prefill, so this is a limit of the single-shot tool path, recorded below. A greedy 8k
generation through the normal chunked path diverged from the dense route's text at
the 15th token, both continuations coherent — expected from non-bitwise math at a
near-tie, and the replay above is the quantified form of that check.

## Not taken now

- **Padding to the largest tile.** S is the largest tile's union in each layer-chunk,
  and the union statistics show the mean tile at half to two thirds of the largest, so
  a third or so of the gathered columns are padding. Two sdpa calls over tiles bucketed
  by union size, or a ragged batched attention, would recover most of it. Unpriced;
  reopen when the attention is measured again on the sparse route (the probe's `sdpa`
  stage now prices this route) and reads above ~20% of the 128k wall.
- **The count readback.** One `n_tiles`-u32 sync per layer-chunk sizes S. A fixed S
  from a bound (the statistics say ~40% of the prefix at T=32) would remove the sync at
  the cost of over-padding. Not priced; the 8k regression is where it would show, and
  the length gate below the crossover covers that instead.
- **Single-shot forwards above ~4k on this route allocate more than the chunked walk.**
  `logits-dump`'s one-forward prefill at 8191 tokens ran out of Metal memory on the
  sparse route and not on the dense one (above). The estimate for n = 8191 is ~5 GB of
  per-layer operands, which should have fit; it did not, and the cause was not chased.
  Reopen if any production path ever forwards more than 2048 tokens at once on a QSA
  layer, or if the replay harness gets a >2048-token fixture and hits this.
- **The 35B's attention is dense and the probe put it at 77-81% of its 128k prefill.**
  That is the flash-at-head-dim-256 ledger item, now with a number.
- **f16 mask straight from the mask kernel** (from the device-mask record) is moot on
  this route: the full f32 mask is read by the tile mask kernel and never converted.

## Review

Codex reviewed the diff read-only and found two real bugs, both at a chunk whose end is
not a multiple of the block ratio — a shape none of the day's benches, the replay or the
first attention test had (131424, 4096 and 2660 all divide by 4):

1. **The tile mask kernel read past the row end for the tail block.** The tail's
   incomplete block has `ratio` gathered columns and up to three of them lie at or past
   `n_kv`; the K/V gather clamped those to row 0, but the mask kernel read
   `mask[row * n_kv + col]` unguarded — the next query's first columns, and past the
   buffer on the last row. Fixed: a column at or past `n_kv` is `-inf`.
2. **The union's bitmap did not know the tail block's id.** The lists name the tail
   block as `n_kv / ratio`, one past the scored `n_blocks`, but the union kernel was
   handed `n_blocks` as its universe: with `n_blocks` a multiple of 32 the id landed in
   a word the bitmap never zeroed or compacted, so the last queries of such a chunk lost
   their own tail silently, and otherwise the union could overflow its row by one.
   Fixed: `attend` passes the ceiling of `n_kv / ratio`.

Both are covered now: `tile_union_matches_the_host_union` puts the last id of a
32-word universe in every row, `tile_gathers_match_host_gathers` names a tail block
whose last column is past `n_kv` and expects row 0 and `-inf` there, and
`sparse_tiles_reproduce_the_dense_mask_route` runs the chunk at `n_kv` 2660, 2661, 2662,
2663 and 3073 (768 complete blocks, a multiple of 32, plus a tail). A Medium — the
gather and query-tiling kernels copy as 4-vectors and validation did not check the
view's alignment — got the check. The reviewer also cleared the barriers, the scan, the
struct ABI (the gather's two `i64` fields after nine `i32` pad identically on both
sides), the sdpa's broadcast mask strides and the un-tiling permutation.

Two things the reviewer noted that stay: padding columns copy K/V row 0 and rely on
it being finite, which is the same reliance dense masked attention has on every masked
column; and the forced greedy replay ran before the length gate existed, so a rerun
needs `XWEN_QSA_SPARSE_MIN_KV=0` on the candidate or a prompt above 49152 tokens.
The 128k and 64k rows above were taken before the two fixes; neither fix changes a
chunk whose end divides by 4, which every one of those prompts has, so the rows stand
for the shipped binary. The Qwen review was skipped at the user's request.
