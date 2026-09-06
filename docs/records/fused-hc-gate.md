# 2026-09-05 — Fused hyper-connection decode gate: 7 dispatches per gate become 3, Flash-Next plain decode 47.0 → 51.2 tok/s (+9% median), the first in-situ confirmation of the dispatch-count lever

Moved verbatim from [docs/log.md](../log.md) on 2026-09-06; the log keeps this entry's
opening paragraph and links here.


The decode budget of the ceiling diagnosis (below) put ~7.4 ms of a 21.3 ms token into
per-dispatch fixed cost at ~4 µs across ~1740 launches, and named the hyper-connection
gates — 672 of those launches, 7 per gate × 96 — as the largest population. This entry
removes 384 of them. Commit dd50397.

**What changed.** Two kernels in `src/ops/hc.metal` replace six of the seven launches a
gate makes at decode (`kernel_hc_write` stays, bit-identical as before):

- `kernel_hc_gate_down` — grouped RMS norm, injection head, down projection and the
  silu(·/hc_count) activation in ONE launch. Grid `(ceil(low_rank/8) + 1, n)`, 160
  threads (five simdgroups); each thread owns two Q8_0 blocks of the 10240-wide carrier
  (blocks `t` and `t+160`, so adjacent lanes read adjacent 34-byte blocks and adjacent
  128-byte runs of the carrier), stages them in registers, and every threadgroup recomputes
  the four per-stream scales itself (10240 cached loads and four simd reductions) rather
  than reading a materialized `normed`. Forty threadgroups each take 8 down rows against
  the staged, normed slice with one reduction pass for all eight; the 41st takes the
  head's four f32 rows. Threadgroup 0 publishes the scales for the second kernel.
- `kernel_hc_gate_up_mix` — up projection, sigmoid and the mean over streams in one
  launch. One thread per (column, stream) with a column's four streams in adjacent lanes
  (`tid = sgid·32 + lane`, structurally), the 320-wide bottleneck staged in threadgroup
  memory, `normed` rebuilt per element from the raw carrier and the published scales in
  the split kernel's expression order, the four-term mean a `simd_shuffle_xor` butterfly.
  Grid `(40, n)` × 256.

Per gate: 3 dispatches (A, B, write) against 7; the tail mixer 2 against 5. Per token:
672 → 288 hc launches, ~1740 → ~1356 in all. Taken when `rows <=
HC_GATE_FUSED_MAX_N` (8, `XWEN_HC_GATE_FUSED_MAX_N`, inclusive), both bottleneck
projections are Q8_0 planes and the geometry passes `hc_gate_fused_supported` (hc_count
a power of two ≤ 8, hidden a multiple of 32, carrier width/32 a multiple of 160 with at
most 2 blocks per thread — production 10240 is exactly 2×160 — low_rank a multiple of
32 up to 1024); anything else keeps the seven-dispatch split path. `XWEN_HC_GATE_CLASSIC`
restores that path outright and is the replay check's `--control` arm; `XWEN_HC_CLASSIC`
still reverts everything to the candle chain. `QLinear::plane()` exposes the raw Q8_0
plane the kernels read. Prefill (rows > 8) is untouched, as the flat prefill column below
shows.

**Why this shape and not the one the ledger priced.** The 2026-08-29 single-threadgroup
norm+head kernel was a 6% decode LOSS (decisions.md "Below 32 tokens the hc norm splits"):
one threadgroup walking the carrier alone leaves the machine idle. Both new kernels keep a
40-41-threadgroup grid and pay for it with redundant, cache-resident reads (each of the 41
threadgroups in A re-reads the 40 KiB carrier row and the 40 KiB norm weight, ~3.2 MB per
gate nominally against 7.16 MB of weights) instead of a second launch. The ledger's
"fused norm+head+down = −192" undercounted: with silu folded into A's epilogue and up+mix
into B, it is −384.

**Rounding.** Both kernels are BOUNDED, not bitwise, against the split path: the 10240-term
down dots and the 320-term up dots are reassociated (per-thread partials, simd_sum, a
5-slot fold). Measured at the real geometry against the frozen `ref_hc` oracle:
fused 1.4e-7 rel_l2 at n = 1 and 1.6e-7 at n = 3. The split path itself is 1.4e-7 at n = 1
but 1.7e-5 at n = 3 — above one token `QMatMul` takes its matmul kernel, whose activation
tiles are half precision — so in the 2..8-token window the fused gate is the MORE accurate
of the two, and the test's cross-path bound is 1e-6 at n = 1 and 5e-5 above it, with the
1e-5 oracle bound doing the holding (`gate_fused_matches_reference`). That window is
where `without_mv_ext` once froze the numerics to the pre-plane chain; the fused gate
moves them, deliberately and behind a kill switch (decisions.md).

**Measurements** (`lowpowermode 0`, automatic; a pinned copy of the built binary,
`--bin`, so no rebuild could touch it).

Flash-Next replay check, `bun scripts/flashnext-replay.ts --control
XWEN_HC_GATE_CLASSIC=1`, oracle reused (pin 6fe7498): code-short 61/64 (3 excused, 0
hard), text-mixed 64/64, long-mixed 59/64 (5 excused incl. one engine near-tie, 0 hard) —
PASS. `cargo test --release`: 1095 passed, 0 failed.

Decode A/B, `generate --no-draft -n 128` on the 596-token `decode-630` fixture, arms
interleaved with the order reversed each round, three rounds, 60 s idle:

| round | fused (default) | classic (`XWEN_HC_GATE_CLASSIC=1`) | pairwise |
|---|---|---|---|
| 1 | 49.2 | 47.0 | +4.7% |
| 2 | 51.2 | 46.7 | +9.6% |
| 3 | 51.3 | 49.0 | +4.7% |
| median | **51.2** | **47.0** | **+8.9%** |

Prefill medians 1175.9 vs 1175.5 tok/s at 596 tokens — unchanged, as it must be. The
bimodal decode the ledger records is visible in both arms (classic 46.7-49.0, fused
49.2-51.3), so the honest range is +5-10% with the median at +9%. The budget's prediction
for −384 dispatches at ~4 µs was ~1.5 ms of 21.3, +7.8%: the measurement lands on it,
which is the first end-to-end confirmation that the residual the ceiling diagnosis
attributed to per-dispatch fixed cost is real and recoverable by launch-count work.

**Review.** Codex and an in-model reviewer on the diff before commit; Codex found no
indexing, barrier or formula error at supported geometry and confirmed `packed_char4`'s
alignment of one makes the 2-byte-offset quant loads legal (MSL spec §2.2.3); its two
validation findings (plane-fits-buffer bound on both launchers; the dump's `hc_gate` label
ignoring `XWEN_HC_GATE_FUSED_MAX_N=0`) are fixed in the commit, and its provenance gap
(no reader enforces the v10 `hc_gate` field, because no graded checkpoint has a
hyper-connection) is ledgered in TODO.md. A Qwen3.8-Flash-Next review after the commit
found no memory-safety or race issue and one real gap — the reference test covered one of
the eight admitted (hc_count, hidden) shapes and n ∈ {1, 3} — closed by
`gate_fused_covers_every_admitted_shape` (both carrier widths at 1, 2, 4 and 8 streams,
plus n = 2 and 8) and `the_fused_gate_ends_at_its_ceiling` (n = 8 fused, n = 9 split,
bitwise), plus four comment fixes; its unmeasured-window hypothesis is ledgered.

**Left on the ledger (TODO.md, decode item 1).** The write folded into the next gate's
norm (−96, the carrier still has to be materialized); `HC_GATE_ROWS_PER_TG` (8) and the
register/occupancy shape of kernel A are unswept — 4 or 16 rows per threadgroup are one-
constant A/Bs; the tail mixer's two launches; and above the 2048 indexer budget the count
is now ~1520-1600, so the QSA tail (60 dispatches) and the MoE glue (−192 possible) are
the next populations.
