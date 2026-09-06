# 2026-08-30 (mm_id tiles) — work-list grid and NR1 64: expert gemms +17-23% in isolation, prefill unchanged end-to-end at 3803 tokens, and the ffn stage turns out to be mostly not the gemms [CONTESTED 2026-09-05: the "mostly not the gemms" reading came off the 2.2x-inflated stage profiler; two in-situ A/Bs bracket the expert gemms at 14-43% of prefill wall, and the tile work reads −2.8% end to end — see "Ceiling diagnosis"]

Moved verbatim from [docs/log.md](../log.md) on 2026-09-06; the log keeps this entry's
opening paragraph and links here.


The prefill-chunk pass left a ledger item for a NARROWER `mm_id` token tile (32 → 16)
on the theory that a 40-row expert wastes most of its second 32-row tile. A code read
refuted it before any bench: `kernel_mul_mm_id_t` dequantizes the expert's whole weight
tile once per TOKEN tile and is dequant-bound, so passes per expert = ceil(rows/NR1),
and a narrower tile RAISES the dominant cost (Flash-Next 1.88 → 2.97 passes at the 2048
chunk; the 35B 2.5 → 4.5). The lever runs the other way: NR1 64 makes those 1.0 / 1.5.
The read also found a bigger waste in the grid: `(t/32, n_out/64, n_expert)` is sized
for one expert owning every row, so at the 2048 chunk ~97% of launched threadgroups
early-return on `r1 >= tpe[e]` (Flash-Next down: 1,310,720 launched, 40,960 useful).

**Design, both behind switches.** (1) Work-list grid: map0 (one thread per expert) now
also exclusive-scans ceil(count/NR1) across the threadgroup (`simd_prefix_exclusive_sum`
+ per-simdgroup totals in the reused shmem; the threadgroup is padded to whole
simdgroups, phantom threads contributing zero and writing nothing) and appends a flat
list `[count, (expert | tile << 16)...]` to the existing scratch. Pass 2 launches
`(ceil(t*top_k/NR1) + n_expert, n_out/64, 1)` — a bound with no readback, since rows
sum to `t*top_k` and each expert adds at most one partial tile — and a threadgroup past
`count` returns; the rest read their pair and proceed exactly as before. All three
families (classic, `_hp`, `_t`) take it; `XWEN_MM_ID_FULL_GRID` restores the old
grid. (2) NR1 templated onto the `_t` family only, `_t64` instantiated for all five
quants; the activation stage becomes NR1/32 sweeps, the store-back tile is 16 KB. The
host picks 64 when the mean rows per expert (`t*top_k/n_expert`) is ≥ 24 (Flash-Next
40, 35B 64 at the 2048 chunk), `XWEN_MM_ID_NR1` forces a width.

**One latent bug surfaced by the width change.** The fork declares the activation tile
`tB` with extents `(NR1, NK)` while `tA` and `tC` are innermost-first and `sb` is
stored NK-contiguous. That is the same shape only while NR1 == NK == 32; at 64 the
tensor read `sb` transposed and the first run landed at rel_l2 1.28 from the oracle.
Now `(NK, NR1)`, with a comment. (`mm.template get_destination_cooperative_tensor`
was the other compile-level consequence of making the kernel NR1-dependent.)

**Bitwise.** All four launch shapes (full/work-list × 32/64, plus the auto-selected
width) are bit-identical on FN gate/up (q4_K, 640×2560) and FN down (q5_1, 2560×640 —
the shipped down dtype, k 640 not being a K-quant multiple) over a 68-expert routing
with row counts {1, 33, 65, 33, 2×64} and over a production-mean routing where the auto
rule picks 64 (`work_list_and_nr1_64_match_full_grid_nr1_32_bitwise`). Measured, not
structural: the two widths are different `matmul2d` instantiations.

**Isolated** (tensor variant, amortized 40 dispatches per sync, t 2048, the
`#[ignore]`d `mm_id_launch_shape_throughput` bench):

| geometry | full-grid NR1 32 (before) | work-list NR1 32 | work-list NR1 64 |
|---|---|---|---|
| FN gate/up q4_K (512E, 640×2560, top_k 10) | 416k tok/s | 466k (+12%) | 512k (+23%) |
| FN down q5_1 (512E, 2560×640) | 202k | 227k (+12%) | 236k (+17%) |
| 35B gate/up q4_K (256E, 512×2048, top_k 8) | 628k | 698k (+11%) | 751k (+20%) |
| 35B down q4_K (256E, 2048×512) | 260k | 276k (+6%) | 281k (+8%) |

**End-to-end** (`generate --no-draft --raw -n 64 --stats`, 3803-token prompt, 60 s
idle between rounds, `pmset -g` before/mid/after: `powermode 0`, no `lowpowermode`
key; prefill / decode tok/s). Flash-Next, two rounds alternated: before
(`FULL_GRID=1 NR1=32`) 841.4 / 43.6 and 799.4 / 42.9; work list at 32: 847.8 / 42.7
and 864.6 / 43.6; default (work list + auto 64): 827.0 / 41.8 and 862.5 / 42.3. Anchors
(1937 tokens, default arm) 1005.8 / 44.6 at the start and 1008.3 / 45.4 at the end, so
no drift. 35B, one round: 2657.2 / 106.1 → 2671.6 / 105.5 → 2708.0 / 106.1. The
arm-to-arm spread on Flash-Next is inside a single arm's round-to-round spread, so **no
end-to-end prefill win is claimable at 3803 tokens**; the 35B's +1.9% is one round and
not separable from noise. Greedy 64-token output byte-identical before/after on both
checkpoints. Gate: 35B ALL PASS, six tiers, digit-identical to 0261e17 (parity.md).

**Where the gain went — attribution, ranking only** (`XWEN_STACK_PROFILE=1`, one run
per arm, sync-inflated so it ranks and does not price). The `ffn` stage fell only 3128
→ 2956 ms on Flash-Next (−5.5%) and 1419 → 1376 on the 35B (−3.1%); every other stage
was flat within ±6%, so nothing absorbed the gain. The expert gemms are a MINORITY of
the `ffn` stage: the rest is the router (softmax + top-k), the f16 rescale chain, the
SwiGLU pass, the combine, and the shared expert (gemms + sigmoid gate), and on
Flash-Next the second chunk is 1755 tokens with a lower mean. A 17-23% cut on that
minority is the 3-5% seen. That finding re-ranks the prefill ledger: the rescale chain
and routing shexp/hc onto `dense_mm` now sit above any further mm_id tile work
(TODO.md). CLAUDE.md "Perf state" carries the one-sentence version.

Review (Opus, Codex, Qwen): no wrong-output bug. Taken: map0's simd scan needed a
threadgroup padded to whole simdgroups (the scan ops are only defined there); host
`ensure!`s for the kernels' 16-bit token/expert/tile indexing (t ≤ 32768 —
reachable only through `XWEN_PREFILL_CHUNK` — n_expert and tile count < 65536) and the
16 KB tile smem; the env width validated on every family; the throughput arms moved
behind `#[ignore]` (they hold ~7.8 GiB of outputs alive). Files: `src/ops/mm_id.metal`,
`mm_id_t_hp.metal`, `mm_id.rs`, `dispatch.rs`, `mod.rs`; `scripts/parity-gate.ts` already
stripped every `XWEN_MM_ID*` name by prefix.
