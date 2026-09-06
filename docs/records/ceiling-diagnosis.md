# 2026-09-05 — Ceiling diagnosis for Flash-Next: achievable bandwidth measured at 537-565 GB/s, a decode token is 57% weight bytes and ~33% per-dispatch fixed cost, prefill is not launch-bound

Moved verbatim from [docs/log.md](../log.md) on 2026-09-06; the log keeps this entry's
opening paragraph and links here.


TODO.md's "FIRST" item asked why decode and prefill sit so far under their ceilings
and to rewrite the perf ledger from the answer. Steps 1-3 (measure bandwidth, decode
budget, prefill budget) are done here; the ledger rewrite is in TODO.md. Machine:
`pmset -g` printed `lowpowermode 0` for every run; the owner says the machine was in
"automatic" battery mode, not high performance. The owner's `xwen serve` was closed
before any GPU run. Entry points: `ops::bandwidth::tests::bandwidth_sweep` (new,
d1ada66), `/tmp/ceil/*.ts` scratch harnesses (disposable), the tensor table from
`xwen inspect` on the UD-Q4_K_XL shards.

**Step 1 — achievable bandwidth (new `bandwidth_sweep`, 5 rounds, 60 s idle, arms
interleaved and order-reversed per round, one sync per arm after `batch`
dispatches).** A reduce-only read kernel and a copy kernel over a 2 GB device buffer,
planes rotating through it so nothing is cache-resident when its turn returns. Medians
of five rounds; the `worst` column is the slowest round:

| arm | batch | µs/dispatch | GB/s med | best | worst |
|---|---|---|---|---|---|
| read 2 GB, 1024 groups | 4 | 4588 | 468 | 516 | 138 |
| read 2 GB, 4096 groups | 4 | 4002 | 537 | 569 | 477 |
| read 2 GB, 16384 groups | 4 | 3976 | 540 | 580 | 535 |
| read 256 MB | 32 | 475 | 565 | 568 | 549 |
| read 32 MB, 4096 groups | 64 | 63.6 | 528 | 535 | 526 |
| read 32 MB, 1024 groups | 64 | 62.5 | 537 | 542 | 501 |
| read 4 MB | 256 | 10.2 | 412 | 434 | 410 |
| read 1 MB | 512 | 4.2 | 250 | 263 | 229 |
| read 256 KB | 512 | 2.7 | 99 | 101 | 91 |
| read 64 KB | 512 | 2.4 | 27 | 28 | 26 |
| read 4 KB, 1 group | 512 | 2.5 | 1.6 | 1.9 | 1.2 |
| copy 1 GB (read+write) | 4 | 4157 | 517 | 532 | 507 |
| copy 32 MB | 64 | 137 | 490 | 531 | 466 |
| copy 4 MB | 256 | 222 | 38 | 508 | 34 |

A single warm round beforehand (no idle) read 572-576 GB/s on the 2 GB and 256 MB
arms, 524-541 at 32 MB, 519 on the 1 GB copy and 511 on the 4 MB copy. Reading:

- **Streaming read is 537-565 GB/s (median), 575-580 in the best rounds: 87-94% of
  the 614 GB/s nominal.** 1024 threadgroups do not saturate the fabric (468); 4096 do.
  Copy runs at ~517 GB/s of bytes touched.
- **A 32 MB plane — the size of `attn_qkv` — streams at 528-537 GB/s in a batch of 64.**
  The shipped Q8_0 gemv reads the same plane at 510 (log 2026-08-30), so the big decode
  gemvs are at 95-97% of what a pure read achieves at that size. Per-kernel bandwidth
  is not where decode's time goes.
- **The per-dispatch floor for a dependent chain inside one encoder is 2.4-2.7 µs** (the
  4 KB / 64 KB / 256 KB arms, whose bytes are free and cache-resident — 512 × 4 KB is
  2 MB — so they isolate the fixed cost). Every read probe writes the same `out` buffer,
  and candle's encoder is `MTLDispatchType::Concurrent` with an automatic memory barrier
  before any dispatch whose buffers overlap a previous output (candle
  `encoder.rs::auto_barrier`), so consecutive probes are barrier-separated exactly the
  way a decode step's dependent dispatches are; the figure is that regime's floor, not
  an independent-dispatch one. The 8.41 µs intercept fitted to the Q8_0 gemv on
  2026-08-30 is therefore mostly that kernel's own ramp and tail, not a launch cost: a
  4 MiB read costs 10.2 µs, of which 7.5 is bytes at 560 GB/s and ~2.7 is fixed. The
  timing brackets host encoding and candle's command-buffer rotation (every 50
  dispatches) as well; at the large arms that is noise, at the tiny arms it is part of
  the cadence being measured.
- The 4 MB copy is bimodal (508 GB/s in one round, 34-38 in four) and unexplained; the
  1 GB and 32 MB copies are not. Noted, not used.
- **High performance mode changes none of it (same evening, owner switched the mode;
  every `pmset -g` from the bench shell then read `lowpowermode 2` where it had read
  `lowpowermode 0` all day, and the owner's own terminal read `powermode 2` at the same
  moment — two names for one key, same value, from different shells; the docs'
  "the key set changed at some point" was this).** Same sweep, 5 rounds: read 2 GB
  562/568 GB/s (4096/16384 groups; 537/540 before), 256 MB 564 (565), 32 MB 516/533
  (528/537), 4 MB 421 (412), tiny-plane floor 2.3-2.4 µs (2.4-2.7), copy 1 GB 511 (517).
  Plain decode at the 596-token prompt 47.3 tok/s median (47.3 / 48.6 / 46.9) against
  47.0; prefill @3851 1139.2 (1137.3 / 1146.5 / 1139.2) against 1140.5. Everything is
  inside the automatic-mode spread except the 2 GB read arms, +4-5%. The mode is not a
  lever for these workloads, and every figure in this entry stands as measured in
  automatic mode.
- The first attempt died before any kernel ran: `Tensor::arange` for f32 builds its
  values on the host by repeated `+= 1.0`, which stops advancing at 2^24, so a 512M
  arange never terminates and grew past the machine's memory until jetsam killed it.
  The buffer is now filled by `Tensor::rand` on the device.

**Byte audit from the tensor table (`/tmp/ceil/bytes.ts` over `xwen inspect`).** The
file is 111.32 GB in 1224 tensors: 77.02 GB of routed experts (144 tensors; gate/up Q4_K
with one Q5_K layer, down Q5_1 with five Q8_0 layers), 28.80 GB PLE table (IQ4_NL), and
5.5 GB of everything else. Weight bytes a decode token reads:

| class | GB per token | note |
|---|---|---|
| gated DeltaNet, 36 layers | 2.247 | `attn_qkv` 27.9 MB + `attn_gate` 16.7 + `ssm_out` 16.7 per layer (Q8_0, 2.206 GB) plus the F32 alpha/beta/conv/norm tensors |
| routed experts, 10 of 512, 48 layers | 1.504 | 3.13 MB per expert slot on average (the quant mix varies by layer) |
| hyper-connection, 96 gates | 0.688 | up/down Q8_0 plus the F32 norm and inject rows |
| `output.weight` (Q8_0, 248320×2560) | 0.675 | |
| full attention, 12 layers | 0.635 | |
| routers (F32) + shared experts + indexer + PLE key/value | 0.577 | indexer weights counted whether or not the layer executes them (≤0.04) |
| **total** | **6.33** | plus GDN state 0.23 GB read+write, KV ≤0.05, indexer keys ≤0.03 |

The "~2.5-3 GB per token" in the TODO item was wrong by 2.2x — it dropped the GDN
projections; the ledger's own 5.5 GB (TODO.md P3 ledger) was nearer: it omitted the
0.58 GB of routers, shared experts, indexer and PLE, and understated its own classes by
~0.25 GB (GDN 2.1 against the audited 2.25, and the rest by rounding). Active matmul parameters per token are
6.67 B (dense 3.64 B, experts 2.36 B, lm_head 0.64 B), i.e. **12.07 GFLOP per prefill
token excluding the lm_head**, against the "~9-10 GFLOP for 4-5 B" the item assumed.

**Step 2 — decode budget on today's binary (`--no-draft --raw -n 128`, 596-token
prompt: 47.0 tok/s median = 21.3 ms/token; 46.0-48.3 across all of the day's plain
runs at this prompt).**

- *Bytes:* 6.6 GB at 537-565 GB/s = **11.7-12.3 ms, 55-58% of the token.** The
  bytes-only ceiling at the streaming rate is **81-86 tok/s**, not the 180-220 the item
  estimated. Priced at each plane size's own measured rate it is lower: the hc (3.5 MB
  planes, ~412 GB/s) and the router/shexp/indexer/attention-k,v planes (1-5 MB, 250-412)
  are ~1.3 GB reading ~1 ms slower than the streaming figure says, so **~75-80 tok/s is
  the bytes-only ceiling with today's plane sizes**, and that ~1 ms belongs to
  small-plane bandwidth, not to the dispatch residual below.
- *Dispatches:* traced from today's code (`/tmp/agent-report-decode-dispatches.md`):
  **1740 compute dispatches per token below the 2048 indexer budget, ~1880-1970 above**
  (1740 − 24 + ~165 = 1881 on most steps, more on the every-fourth step that rebuilds
  the block keys), and
  **3 host syncs** — the token-id readback before the layer loop
  (`stack.rs:511`, ids the host uploaded one line earlier), the batched PLE readback
  mid-layer 1 (`ple.rs:1492`), and the sampler's logit row. By population:
  hyper-connection gates + writes 672 (7 per gate, 14 per layer — decode takes the split
  arm below 32 tokens, so the injection head is its own dispatch), MoE 576 (12 per
  layer), GDN 252 (7 per layer since the beta|alpha fold — the ledger's 288 is stale),
  full attention 192-216, QSA indexer 24 below budget / ~165 above, PLE 11, embed and
carrier seed 6, tail 6. The
  ledger's "~1000 dispatches/token" was understated 1.75x; the missing mass is the hc
  carrier, the largest population in the model.
- *Fixed cost bracket:* 1740 × 2.5 µs = 4.4 ms at the tiny-kernel floor; 1740 × 8.4 =
  14.6 ms at the gemv intercept. What closes the budget: 21.3 − 12.0 (bytes, which
  already include the scan's 0.23 GB of state) − 0.9 (3 syncs at ~0.3 ms each, from the
  readback batching's +2.85% for two removed waits) − 1.0 (the serial scan's amortized
  1.4 ms less its own bytes; its 36 launches are inside the 1740) ≈ 7.4 ms ≈ **~4 µs of
  fixed cost per dispatch on average**. That average is a RESIDUAL ATTRIBUTION, not an
  independent measurement: the independent anchors are the 2.5 µs tiny-kernel floor
  and the 8.4 µs gemv intercept, and the residual lands between them, as it should for
  a mix of tiny glue kernels and gemvs. A decode step is mostly a dependent chain (each
  dispatch consumes the previous one's output), so candle's automatic barriers
  serialize most of it; the independent pairs (q/k/v, gate|up) can overlap and are
  part of why the average sits below the gemv intercept. One more caveat on the floor
  itself: the probe brackets host encoding, and 1740 × 2.4 µs ≈ 4.2 ms is close to the
  3.7 ms of process CPU a decode token costs, so the 2.4-2.7 µs may be the CPU's encode
  cadence rather than GPU drain-and-fill; the probe cannot separate the two. The test
  that would: process CPU time of the tiny arm against its wall (CPU ≈ wall means the
  floor is encode-side, and candle's per-dispatch locking — TODO.md survey item — is the
  lever).
- *Not CPU-bound in the process:* `/usr/bin/time -l` differenced between 32 and 198
  decoded tokens (the `-n 256` runs stopped at 198; two pairs, user 1.89/1.91 →
  2.38/2.38 s, sys 8.24/8.23 → 8.36/8.42) gives **2.95 and 2.83 ms user + 0.7 and 1.15
  ms sys per token, a 17-19% CPU duty cycle.** The main thread spends most of the token
  waiting on the GPU. This excludes the process's own CPU as the bottleneck; it does
  not see driver or kernel work outside the process.
- *Not command-buffer-bound:* `CANDLE_METAL_COMPUTE_PER_BUFFER` 50 (default) / 250 / 10
  read 47.0 / 46.7 / 46.3 tok/s medians over three interleaved rounds — 5x fewer or 5x
  more command buffer commits per token move nothing outside noise. Candle keeps one
  compute encoder open across dispatches (concurrent dispatch type, automatic barriers
  between dependent ones); the cost is per dispatch, not per commit.
- *Stage ranking (sync-inflated, rank only; decode is 11.3 tok/s under the profiler):*
  `ffn` 20.7%, `mixer_delta` 19.3%, `ffn_norm` 13.3%, `attn_norm` 13.0%,
  `residual_attn` 10.1%, `residual_ffn` 9.7%, `mixer_full_attn` 5.4%, `ple` 3.8%,
  `qsa_select` 2.4%, `lm_head` 1.5%. The four hc stages together rank at 46%, which is
  the 672-dispatch population showing through the bracket inflation.

**Decode diagnosis.** At the measured bandwidth the token is 57% bytes and ~33%
residual attributed to per-dispatch fixed cost in a mostly dependent chain, with ~4%
syncs and ~5% serial scan beyond its bytes. Nothing is
"far below peak": the gap to the bytes-only ceiling (81-86 tok/s) is the dispatch count.
Realistic levers, priced at ~4 µs per dispatch removed: halving the count (−870) is
−3.5 ms → ~56 tok/s (+9); a whole-block fusion that brought it near 400 would be
~15.9 ms → ~63 tok/s. The hc carrier (672) is the first population to attack, MoE (576)
second; the three GDN fusion candidates the ledger prices at +1-2%, +1-2% and +2-4% at
8.41 µs are ~0.15 ms each for the two −36 ones at 4 µs (≈0.7%) and ~0.3 ms for the −72
one. The token-id readback sync is free to
remove (~0.3 ms, +1.4%).

**Step 3 — prefill budget at the 2048 chunk (3851 tokens: 1129-1140 tok/s = 3.38-3.41 s;
880 tokens: 1262).**

- *FLOPs:* 3851 × 12.07 GFLOP = 46.5 TFLOP → **13.7 TFLOP/s achieved end to end** (the
  item said 10-11). Matmul FLOPs only — attention scores and the GDN scan are excluded —
  so the achieved figure is a lower bound. Gemm-only ceiling at the 28-36 TFLOP/s the dense gemm reaches in
  isolation: 2300-3000 tok/s.
- *Weight bytes:* at 2048 rows per chunk every expert is touched (2048 × 10 / 512 ≈ 40
  rows each under uniform routing, and any expert with one row costs its full weight
  read; real routing is skewed but leaves an expert untouched only rarely), so each
  chunk reads the whole 82.5 GB trunk once: 165 GB for two chunks = **0.30 s at 550
  GB/s, 9% of wall — a LOWER BOUND on the gemm time, not a term to add to it**: the
  expert gemm's weight reads are inside its measured time. It is also part of why 2048
  beat 512: per 2048 tokens, four 512-row chunks read the trunk four times (330 GB
  against 82.5).
- *Expert gemms from the amortized bench (`mm_id_launch_shape_throughput`, t=2048, 40
  dispatches per sync, log 2026-08-30 "mm_id tiles"):* gate/up 512k tok/s = 4.0 ms per
  plane per 2048 rows, down 236k = 8.7 ms → 16.7 ms per layer per chunk → 48 layers ×
  3851/2048 rows ≈ **1.5 s ≈ 44% of the 3.41 s wall**, i.e. the expert gemms run at
  ~12 TFLOP/s (18.2 TFLOP of the 46.5), dequant-bound by the ledger's 2026-08-30 code
  reading (not re-measured here). This
  CONTESTS the 2026-08-30 conclusion that the gemms are "a minority of `ffn`", which
  was drawn from the sync-inflated stage profiler plus a noisy end-to-end. The two
  in-situ A/Bs below bracket the share at 14-43%.
- *Dispatch floor:* ~1650 dispatches per chunk (the decode count less the split arm's
  96 injection dispatches; no separate prefill count was taken) × 2 chunks × 2.5-8 µs =
  8-26 ms, **under 1% of wall.** Prefill is not launch-bound at any plausible count.
- *Host side NOT cleared for prefill:* differenced 880 → 3851 tokens, 0.11 ms user +
  0.5 ms sys per prefill token against 0.885 ms wall — a 68% process duty cycle,
  sys-dominated (the process sits at ~108 GB resident on a 128 GB machine, so page
  traffic on the mmapped weights is the likely source). Decode's 17% clears the host;
  prefill's does not, and whether that sys time is on the critical path is an open
  question the duplicate-dispatch probe would also answer.
- *Stage ranking (`XWEN_STACK_PROFILE`, rank only — it read 511 tok/s against 1129
  unprofiled, a 2.2x inflation, so it does not price prefill either, contrary to what
  this session first assumed):* `ffn` 32.2%, `mixer_delta` 20.3%, `mixer_full_attn`
  12.7%, `ffn_norm` 8.7%, `attn_norm` 7.5%, `residual_attn` 7.3%, `residual_ffn` 6.9%,
  `qsa_select` 2.9%, `ple` 0.9%. At 880 tokens the same shape (`ffn` 36%).
- *hc activation traffic, estimated:* the carrier is [2048, 10240] f32 = 84 MB per chunk;
  a gate reads and writes it ~8-10 times across norm, down, mix and write → ~0.7-0.8 GB
  per gate, ~67-77 GB per chunk over 96 gates → **~0.25 s per 3851-token prompt (~8%)**
  at the streaming rate. An estimate, unmeasured in situ.

**In-situ A/Bs on the expert gemms (same day, `decode-arms.ts`-style interleaved
rounds, 60 s idle, 3851-token prompt, `-n 4`, `lowpowermode 0` before and after).**
The amortized bench was re-run on today's binary with a fourth arm, the classic f32
(`_hp`) tile family at its default tuning, so that a wall-clock A/B against
`XWEN_MM_ID_CLASSIC=1` converts into a gemm share (df14d7e):

| geometry, t=2048 | full-grid NR1 32 | work-list NR1 32 | work-list NR1 64 (shipped) | classic-hp |
|---|---|---|---|---|
| FN gate/up q4_K | 422k tok/s (4.85 ms) | 483k | 525k (3.90 ms) | 226k (9.05 ms) |
| FN down q5_1 | 214k (9.55 ms) | 242k | 251k (8.17 ms) | 157k (13.08 ms) |
| 35B gate/up q4_K | 711k | 820k | 834k | 446k |
| 35B down q4_K | 299k | 316k | 320k | 247k |

(ms = per plane per 2048-row chunk.) Shipped: 2 × 3.90 + 8.17 = **15.97 ms per layer
per chunk**; classic 31.18; old full-grid NR1 32 19.25. Over 48 layers × 3851/2048 rows
(90.3 layer-chunks) the amortized figures predict **1.44 s of expert gemm** in the
shipped arm, +1.37 s under classic tiles, +0.30 s under the old full grid.

| arm | rounds | prefill tok/s | wall s | Δ wall vs base |
|---|---|---|---|---|
| base (shipped tiles) | 4 | 1143.5 / 1140.8 / 1141.5 / 1150.1 → med 1143.5 | 3.368 | — |
| `XWEN_MM_ID_CLASSIC=1` | 4 | 866.0 / 856.9 / 858.0 / 852.8 → med 858.0 | 4.488 | **+1.12 s (+33%)** |
| `XWEN_MM_ID_FULL_GRID=1 XWEN_MM_ID_NR1=32` (pre-2026-08-30 tiles) | 3 | 1140.5 / 1141.5 / 1134.4 base → 1109.8 / 1109.0 / 1109.5 | 3.471 vs 3.377 | **+0.094 s (+2.8%)** |

The two A/Bs do not agree on how the isolated rates transfer. The classic arm's
+1.12 s is 82% of the +1.37 s the amortized rates predict; the full-grid arm's +0.094 s
is 32% of its predicted +0.30 s (and it is a real, reproducible −2.8%, where the
2026-08-30 session saw "nothing claimable"). Solving each A/B for the shipped gemms'
in-situ time under the assumption that each kernel family's isolated ratio transfers:
the classic A/B implies **~1.18 s (35% of the 3.37 s wall)**, the full-grid A/B implies
**~0.46 s (14%)**, and the raw amortized figure is 1.44 s (43%). So the expert gemms'
share of prefill wall is **bracketed at 14-43%** — the largest PRICED candidate (GDN
prefill, ranked second by the profiler and unpriced, could be comparable at the low
end), but not settled, and the 2026-08-30 "gemms are a minority of `ffn`" reading is
CONTESTED rather than refuted: that reading came off the 2.2x-inflated stage profiler,
whose inflation falls hardest on the many-small-dispatch glue stages and so understates
the gemms, while the full-grid A/B says the work-list/NR1-64 advantage measured on
uniform synthetic routing mostly does not survive real routing. Two caveats cut the
other way and apply to both arms: the classic arm is NOT a pure gemm swap — `ClassicHp`
is outside `casts_activation_f16` (`ops/mod.rs`), so `needs_rescale` is false and the
classic arm takes the fused MoE epilogue the shipped f16-tile arm cannot (`moe.rs`),
which makes its glue cheaper and its +1.12 s an UNDERSTATEMENT of the gemm delta (the
share implied by that arm is ≥35%, not =35%); and the uniform-routing discount that
explains the full-grid arm applies equally to the 43% upper bound, which comes from the
same uniform-routing bench — a bench that, unlike the A/Bs, runs one 40-dispatch pass
per arm with no rounds or idle. The instrument that
settles it is an in-situ duplicate-dispatch probe (encode a stage's kernels twice
behind a switch, read the wall delta), ledgered in TODO.md as the next step; neither
the stage profiler nor an isolated bench can.
