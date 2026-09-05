# Engineering log

Reverse-chronological. Heading convention: `## YYYY-MM-DD — headline stating what
shipped, ideally with the number`. Same-day entries disambiguate in the heading text.
Superseded entries are marked in the headline, never deleted.

## 2026-09-05 — Duplicate-dispatch probe prices Flash-Next prefill in situ: expert gemms 1.09 s of 3.42 s @3851 (32%), MoE glue 0.40, hc gates 0.39, GDN 0.23, shared expert ~0

The instrument the ceiling diagnosis asked for (below). `XWEN_DUP_STAGE=<names>` makes
the multi-token path encode a named stage's kernel launches `XWEN_DUP_REPS` extra times
(default 1) with the copies' results dropped, so the prefill wall delta against a plain
run is that stage's GPU time in situ — no syncs, no host work duplicated, only launchers
that are pure functions of tensors the caller already holds (`ops::dup`; the router,
gathers, cache advances and allocations of the surrounding blocks run once). Stages:
`experts` (the three mm_id gemms), `experts_down`, `moe_glue` (router, activation,
epilogue), `shexp`, `hc` (norm, both bottleneck gemms, silu, mix, write), `hc_gemm`,
`gdn` (conv, beta/decay head, scan, gated norm), `gdn_scan`. Commit ab43499; parity.md
carries the switch row. The delta is a LOWER bound for a stage that leaves the GPU idle
(candle's encoder barriers only on buffer hazards, so a copy writing a fresh buffer may
overlap the original); the two-copy experts arm below reads 1.03 s per copy against 1.09
for one, so at least that stage has little overlap to hide.

**Protocol.** `xwen generate --no-draft --raw -n 4 --stats` on the 3851-token
`prefill-4k` fixture, Qwen3.8-Flash-Next at the 2048 chunk, ten arms interleaved with the
order reversed every round, three rounds, 60 s idle between rounds, medians;
`pmset -g` said `lowpowermode 0` (automatic) throughout. The binary was a detached
worktree build of ab43499 at /tmp/xwen-bench — an earlier attempt on the main tree's
binary aborted mid-session because a coding agent's `cargo build` had swapped
`target/release/xwen` (and its `include_str!` kernels) under the harness; a first,
unpinned session of five arms is quoted only as a replicate.

| arm (stage duplicated once) | tok/s | wall s | delta s | share of 3.42 s | unpinned replicate |
|---|---|---|---|---|---|
| base | 1126.4 | 3.419 | — | — | 1131.1 (3.405 s) |
| experts (gate, up, down mm_id) | 854.3 | 4.508 | 1.089 | 31.8% | 0.959 (28.2%) |
| experts, two copies (`XWEN_DUP_REPS=2`) | 703.7 | 5.473 | 2.054 = 1.027/copy | — | — |
| experts_down | 987.2 | 3.901 | 0.482 | 14.1% | 0.423 (12.4%) |
| moe_glue (router, activation, epilogue) | 1009.8 | 3.814 | 0.395 | 11.5% | — |
| shexp (shared expert, both gemms) | 1123.3 | 3.428 | 0.009 | 0.3% | — |
| hc (norm, down, silu, up, mix, write) | 1011.5 | 3.807 | 0.388 | 11.3% | 0.346 (10.2%) |
| hc_gemm (the two bottleneck gemms) | 1082.8 | 3.557 | 0.138 | 4.0% | — |
| gdn (conv, beta/decay, scan, gated norm) | 1055.9 | 3.647 | 0.228 | 6.7% | 0.222 (6.5%) |
| gdn_scan | 1077.2 | 3.575 | 0.156 | 4.6% | — |

Round-to-round spread on the duplicated arms was under 1% except the experts arms
(±2-4%); the 4-token decode column is noise at that length and is not quoted.

**What it settles.**

- **The expert gemms are 28-32% of prefill wall (0.96-1.09 s of 3.4 s), inside the
  ceiling diagnosis's 14-43% bracket** and closer to the amortized bench's 1.44 s than
  to the full-grid A/B's 0.46. Gate+up together are 0.61 s (18%), down 0.48 s (14%) —
  the down plane is 44% of the expert time, not "about half". The 0.30 s of weight
  re-reads sits inside this.
- **The 2026-08-30 "gemms are a minority of `ffn`" reading is REFUTED.** In situ the
  `ffn` stage is experts 1.09 + glue 0.40 + shexp 0.01 = 1.50 s (44% of wall) and the
  gemms are 73% of it. The glue is real — 0.40 s, as much as every hc gate together — but
  it is the smaller half, and the shared expert is free at prefill (two dense gemms at
  2048 rows, 0.3%).
- **The hyper-connection gates are 11% (0.39 s), of which the bottleneck gemms are 4%
  and the four glue kernels plus the write 7% (0.25 s).** The ledger's "~8% activation
  traffic" estimate was about right.
- **GDN kernels are 7% (0.23 s): the chunked scan 4.6%, conv + beta/decay + gated norm
  2.1%.** The stage profiler's 20% for `mixer_delta` includes its three projections,
  which the probe does not cover.
- **Priced: 2.11 s = 62% of wall. Unpriced 1.31 s (38%):** the GDN and attention
  projections, full attention and QSA selection (12 layers), PLE, embedding, lm_head, and
  whatever the concurrent encoder hides between stages. Those are the next probe stages if
  the ranking needs them; attention and QSA mutate caches around their kernels, so the
  wrap has to sit inside the block.

**How the ledger reads now (TODO.md, prefill).** Expert gemm efficiency stays first at a
measured 0.96-1.09 s; MoE glue (0.40 s) and the hc glue (0.25 s) are second and third and
are now priced rather than estimated; the GDN scan (0.16 s) is fourth; the shared expert
drops off the list. Every one of these is a fusion or kernel-efficiency lever, none is a
launch-count lever (prefill dispatch floor <1%, unchanged).

## 2026-09-05 — Ceiling diagnosis for Flash-Next: achievable bandwidth measured at 537-565 GB/s, a decode token is 57% weight bytes and ~33% per-dispatch fixed cost, prefill is not launch-bound

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


## 2026-09-05 — PLE gate and conv move to device for multi-token prefill: Flash-Next prefill +12-13% at 880 and 3851 tokens, on by default (`XWEN_PLE_TAIL_CLASSIC` restores the host tail)

Picked up from TODO.md's "Next Flash-Next perf work" item (P3 item (5)), started in a
Codex session and finished here. `XWEN_PLE_PROFILE` put the host gate plus conv at
41 + 198 ms per 2048-token chunk (213-238 ms across two profiles, `powermode 0`), and
at decode the pair costs ~0.13 ms/token, which nobody has turned into a qualified gain,
so decode keeps the host tail and the batched readback of the entry below.

**What shipped.** Two Metal kernels in `src/ops/ple.metal`, dispatched by
`ops::dispatch::run_ple_tail` through `ops::ple::ple_tail`. `kernel_ple_gate` runs one
256-thread threadgroup per (token, stream): grouped RMS norm of the carrier × the query
norm weight, the key·query dot scaled by 1/√2560, signed sqrt with the 1e-6 clamp,
sigmoid, gate × the shared value, then the gated row's own grouped norm × the conv norm
weight. `kernel_ple_conv` is one thread per output element: the four dilated taps,
reading the uploaded channel-major history for positions before the chunk, then
`gated + silu(acc)`. The reductions partition the oracle's sums across simdgroups (f32,
against the host's f64 sum of squares); every scalar product keeps its order. Safe math
mode, not the fast mode of every other library here, so the gate's `isnan` guard and the
oracle's NaN-propagation contract survive. On for Metal forwards with `n > 1`;
`XWEN_PLE_TAIL_CLASSIC` (presence-based like the other kill switches) restores the host
tail; decode and CPU forwards never take the kernels.

The conv state stays host-owned and its layout is untouched: after the kernels,
`ops::ple::readback_tail` blits back only the last `min(n, 9)` normalized rows (all `n`
rows while a checkpoint is armed, so a partial accept can land between tokens) and
`PleLayer::commit_device_history` rebuilds the window, the per-token trail and the
n-gram history exactly as the classic path does. Snapshot, rollback, image restore and
the on-disk tier see the same bytes. A forward that would overrun an armed span now
fails before touching the state on both paths (`record` used to refuse it after the
conv window had already advanced).

**Verification.**

- Unit tests (23 pass, `cargo test --release --lib -- 'qwen4exp::ple::' 'ops::ple::'`):
  the kernels against the frozen `PleLayerRef` oracle at synthetic and real (4×2560,
  k=4, dilation 3) geometry, chunked history continuation, gate zero/near-zero/NaN,
  offset operand views, 2048-token device-vs-classic agreement at ≤1e-5 on the addend
  and the state, checkpoint commit 0/partial/full with image restore and decode
  continuation, and the rejected-overflow-leaves-state-unchanged case.
- Real inputs: a temporary hook ran both tails on the same 612-token prefill of the
  long-mixed fixture. Device vs classic: addend max abs 5.96e-7, rel L2 7.95e-7, zero
  elements off by more than 1e-3, conv state max abs 8.1e-6, histories equal.
- Determinism: the device arm's 64-step replay is bitwise identical across three runs
  and two builds (before and after the review edits below).

**Forced replay against llama.cpp, and why the hard mismatch is not the kernel.** The
Codex session graded the arm with the Flash-Next forced-replay protocol (parity.md;
64 steps × 3 fixtures, q8 margin 1.0, ≤8 excuses) and got code-short 62/64 (2 excused),
text-mixed 64/64, long-mixed 59/64 with 4 excused and **one hard mismatch at step 4**
(oracle 8214 at a 1.829-logit margin over the candidate's 35935). Classic on the same
binary and oracle: 63/64, 64/64, 62/64 with 0 hard. So a control was run on
long-mixed with the device path OFF and exactly one change to the classic host code:

| long-mixed arm | agree | excused | hard |
|---|---:|---:|---|
| classic | 62/64 | 2 | 0 |
| device kernels | 59/64 | 4 | step 4, margin 1.829 |
| classic, key·query dot summed in reverse order | 60/64 | 3 | **step 4, margin 1.829** |
| classic, key·query dot accumulated in f64 | 61/64 | 3 | 0 (but 4 winners change vs classic, common top-5 logits move up to 1.37) |

Reversing the summation order of one f32 dot product, nothing else, reproduces the same
hard mismatch at the same step. A strictly more accurate f64 dot passes the letter of
the gate and still moves logits by more than a unit. Classic is bitwise deterministic run
to run, so every difference is the perturbation. The long-mixed step-4 decision is
chaotic at ulp scale on this 512-expert, top-10 checkpoint: classic itself has 8214 at a
0.31 margin over 35935 where the oracle has 1.83. The gate cannot resolve this change,
and the direct real-input comparison above is the instrument that can. Recorded in
parity.md as a limitation of the Flash-Next stand-in gate. Scripts and dumps under
`/tmp/ple-ctl/` (disposable).

**Bench** (`XWEN_BENCH=1 generate --model-size flash-next --no-draft --raw -n 64 --stats`,
the committed `prefill-4k.txt` = 3851 tokens and `prefill-925.txt` = 880 tokens, arms
interleaved C D D C then D C C D, 60 s idle between runs, no other GPU work; `pmset -g`
before and after: `lowpowermode 0`, and this session the `powermode` key was absent
again, so still no high-power claim):

| prompt | classic prefill | device prefill | decode C / D |
|---|---|---|---|
| 3851, round 1 | 997.9 / 1009.1 | 1132.2 / 1133.7 | 45.9, 45.9 / 45.3, 44.7 |
| 3851, round 2 | 1010.0 / 1010.4 | 1139.7 / 1142.9 | 45.7, 45.6 / 45.3, 46.1 |
| 880, one round | 1126.5 / 1108.7 | 1263.9 / 1259.1 | 45.9, 46.4 / 48.3, 47.3 |

Medians: **1010 → 1139.7 at 3851 (+12.8%)**, **1117.6 → 1261.5 at 880 (+12.9%)**.
Decode is the same path in both arms and reads flat. The classic cells drift +1.3%
across the 3851 rounds, inside the 3% flag. The saving matches the profile: ~230 ms of
host work per 2048-token chunk (the isolated tail transaction went 236-238 → 6-7 ms in
the Codex session's microbench) against a ~3.8 s prefill.

**Review.** Codex (gpt-6-astra), a same-model reviewer and Qwen3.8-Flash-Next all read the
diff; none found a correctness defect in the math, the indexing or the state commit.
Applied from the reviews: unsigned loop counters in the gate kernel (the signed ones
overflowed only at `hidden` near 2^31, unreachable through `PleLayer`), `partial[32]`
sized for the most simdgroups a threadgroup can hold instead of the 8 the 256-thread
launch happens to use, a comment on why the `gated` scratch may drop before GPU
completion (private pool, encoder-tracked read-after-write, no CPU writer), a
`checked_mul` in `readback_tail`, `XWEN_PLE_DEVICE` added to `parity-gate.ts`'s stripped
env, and doc comments that had lost their referents. Left alone: the armed-path window
rebuild is a strided gather per token (unreachable until Flash-Next has a drafter), the
`exp` per element in the conv kernel under safe math is unpriced, and the three device
weight copies (~240 KiB) are built eagerly.

**Default flipped the same day, and the check codified.** It first landed opt-in
(`XWEN_PLE_DEVICE`, 561635f) because the replay stand-in failed as written; the owner
called the flip and asked that the correctness check stop failing on this class of step.
`scripts/flashnext-replay.ts` now owns the Flash-Next forced replay (oracle build/reuse,
candidate and `--control` arms, grading), with one rule added to the decode tier's
excusal: a mismatch is also a near-tie when the control arm, the same binary with the
change switched off, holds the oracle's token over the candidate's pick by less than the
band. The oracle band said "the reference was not sure"; this says "the engine was not
sure either", which is the only way ulp-level reassociation can move the answer. The
≤8 cap and the hard rule for everything else stand. Run with the new default and
`--control XWEN_PLE_TAIL_CLASSIC=1` against the cached llama.cpp oracle (pin 6fe7498):

| fixture | agree | excused (step: side, margin) | hard |
|---|---:|---|---:|
| code-short | 62/64 | 23 oracle 0.034, 36 oracle 0.694 | 0 |
| text-mixed | 64/64 | | 0 |
| long-mixed | 59/64 | 3 oracle 0.705, **4 engine 0.313** (oracle 1.829), 12 oracle 0.435, 27 oracle 0.141, 49 oracle 0.138 | 0 |

**PASS**, 185/192 agreeing, 7 excused, 0 hard, the same 185/192 the 2026-08-30 fold read
on its day. Step 4 is the one the new rule exists for; nothing else changed side.
Decode tail stays on the host (no qualified gain at 0.13 ms/token); see TODO.md.

## 2026-09-05 — PLE batches its three device-to-host readbacks

The next Flash-Next perf candidate selected from TODO.md was the PLE readback
collapse. `PleLayer::forward` previously called `to_vec1` separately for key, value
and carrier, and candle's `MetalStorage::to_cpu` allocated, blitted and waited EACH
time. At single-token decode, `readback_inputs` now encodes all three copies into
one shared staging buffer and waits once. Multi-token prefill keeps its original
independent transfers. The key and carrier are 10240-wide, the value 2560-wide: 90 KiB/token
total. The host gate, conv, recurrent state, table gather and frozen references are
unchanged. `XWEN_PLE_READBACK_CLASSIC=1` restores the old transfers. The ownership,
offset and temporary-memory choices are in decisions.md "PLE decode readbacks".

**Verification:** 17 tests pass with
`cargo test --release --lib qwen4exp::ple::tests -- --test-threads=1`, including
the new bitwise readback comparison at one token and the full 2048-token chunk, unequal plane widths, offset/strided inputs,
F16 conversion, empty inputs and CPU fallback. The existing tests cover the fixture,
real geometry, chunk continuation, checkpoint/rollback and image state. The empty
F16 test initially hit candle's divide-by-zero dispatch for an empty conversion; the
helper now skips empty planes before conversion or blitting.

`bun scripts/parity-gate.ts`: **ALL PASS (6 graded)** on the 35B, with the previous
record's exact summary: strict cosine 1.000000, mm 0.999618; decode 63/64, 62/64,
61/64 with 1/2/3 excused and zero hard mismatches; ppl Δnll 0.001179. Its disposable
`/tmp` reference caches needed rebuilding (21 s full-logit, 74–83 s per greedy
fixture); the committed ppl reference was reused and no floor or reference code
changed. Flash-Next still cannot use that harness (parity.md's standing limitation).
Instead, the old readback arm free-ran 64 steps on each of code-short/text-mixed/
long-mixed and the new arm replayed those histories: **192/192 exact**, including
all top-five logit values, L2 norms and nonfinite counts. This is equivalence to the
pre-change engine, not a fresh llama.cpp comparison.

**Initial all-length experiment:** 18 full-model runs: four AB/BA rounds over
612 and 3677 tokens, plus start/end classic anchors at 612. Both the arm order and
length order reverse each round, with 60 s idle before each round and both anchors.
The initial candidate predates the final `n != 1` guard at the readback call;
omitting that guard reproduces the all-length experiment.
Every run uses `XWEN_BENCH=1 generate --model-size flash-next --no-draft --raw
--temp 0 --top-k 1 -n 128 --stats --max-ctx 8192`, with the committed long-mixed
fixture once or six copies joined by two newlines. No concurrent GPU work. Power
before/mid/after: `powermode 0`; no high-power claim. All outputs are byte-identical
between arms at each length. Metrics were routed to a separate file.

Rates below are tok/s; C = classic independent reads, B = all-length batching:

| prompt | round/order | prefill C | prefill B | decode C | decode B |
|---|---|---:|---:|---:|---:|
| 612 | 1 / CB | 1098.2 | 1098.4 | 45.36 | 46.49 |
| 612 | 2 / BC | 1098.7 | 1088.4 | 45.92 | 46.98 |
| 612 | 3 / CB | 1062.6 | 1080.5 | 45.81 | 45.16 |
| 612 | 4 / BC | 1104.0 | 1098.1 | 45.19 | 47.03 |
| 3677 | 1 / CB | 1011.5 | 967.2 | 44.88 | 45.56 |
| 3677 | 2 / BC | 974.0 | 988.6 | 44.80 | 46.35 |
| 3677 | 3 / CB | 997.2 | 967.5 | 44.69 | 45.06 |
| 3677 | 4 / BC | 1000.3 | 1003.6 | 45.13 | 46.12 |

Median decode rises 2.5% short and 2.2% long, but one short pair loses. Prefill
medians read −0.5% short / −2.1% long and the long-prompt pair ratios depend on
order. **Drift flag:** the decode anchor is 46.29 → 47.67
(+3.0007%, just over 3%); its prefill is
1106.4 → 1068.0 (-3.47%). Keep every cell, including
the loss; this is directional evidence for decode, not a clean absolute-throughput
headline. **Decision: enable batching only at seq == 1.** The wait cost matters
per token, while prefill has no established gain and its combined staging allocation
is larger. Multi-token prefill therefore keeps the existing path.

The ignored `ple_readback_bench` independently measures the complete transfer
transaction (including queued producers), with alternating arms and a wait per
transaction because a CPU readback requires completion. Two probes: decode
0.50–0.59 → 0.17–0.23 ms; 512-token reads approximately level; 2048-token reads
13–14 → 11–12 ms. These are isolated transfer costs, not per-step profiler charges
and not end-to-end throughput estimates.

**Final shipped configuration, decode only:** the 17 PLE tests and all six
35B parity checks pass again. A fresh 3677-token CB / BC check uses the same
128-token commands, a 60 s idle before each pair, and the first/last classic runs
as its start/end anchors. `powermode 0` before and after. Results:

| order | prefill classic | prefill decode-batched | decode classic | decode batched |
|---|---:|---:|---:|---:|
| CB | 992.1 | 988.7 | 44.99 | 46.23 |
| BC | 986.3 | 1005.9 | 44.57 | 45.89 |

Decode **+2.74% / +2.95%**, central rates **44.78 → 46.06 tok/s (+2.85%)**.
Classic decode anchor drift −0.94%, prefill −0.59%, both below the 3% flag. Prefill
varies −0.34% / +1.99% despite running the exact same transfer path, so no prefill
gain is claimed. Every run emits the same 128 tokens, byte-identical to the initial
sweep's long-prompt output. The decode-only route is the shipped default;
`XWEN_PLE_READBACK_CLASSIC=1` is its control. This measures `generate`; serve uses
the same PLE forward but was not independently benchmarked in this session.

**Candidate screened first, left unchanged:** hc's two Q8_0 decode projections on
the existing vendored `matmul_q8`. The new ignored `hc_q8_decode_bench` rotates 96
distinct shared QMatMul/kernel allocations (334 MB) at each actual shape, retains
outputs through the batch flush, and alternates arm order. Two exploratory probes
were too noisy to qualify the down projection; the up projection lost in both
probes (the vendored kernel launches 5120 two-row threadgroups at k=320). No route
changed and no end-to-end gain is claimed for that candidate. The bench remains available
for a future shape change, and the ledger annotation preserves the unresolved down
projection.

Artifacts for this session: `/tmp/xwen-ple-perf/` (replay JSONs, gate/build/test and
benchmark logs, runner scripts, raw per-run metrics); the exploratory Q8 logs are
`/tmp/xwen-hc-q8-bench{,2}.log`. These are disposable; the protocol and measured
results are recorded here so their later cleanup does not erase the result.

## 2026-09-05 — Shared agent instructions live in AGENTS.md

Moved the root agent context from `CLAUDE.md` to `AGENTS.md`, preserving the
instructions. `CLAUDE.md` imports it with `@AGENTS.md` so both entry points use the
same source. Updated the README and justfile references to the canonical file.

## 2026-09-05 — Per-run metrics on disk and `xwen stats`: every surface appends a JSONL record, aggregated by day/model/surface/session

**Context.** Nothing in xwen remembered a run after it finished. The `--stats` line and
the API usage objects report a run to whoever asked for it and then the numbers are
gone, so questions the machine should be able to answer — how much did this week cost,
which checkpoint did the work, is serve's prefix reuse actually hitting, which Claude
Code session spent the afternoon — had no source but scrollback. This arc gives every
surface a place to write one line and gives that file a reader.

**What ships.** `src/metrics.rs` plus a `stats` subcommand. Every finished run appends
one JSON object to `$HOME/.local/state/xwen/metrics.jsonl`: schema version, completion
timestamp, surface, checkpoint name, prompt/cached/prefill/decode tokens with the
seconds each phase took, `ok`, and the optionals a surface may know (thinking tokens,
drafted and accepted positions, batch items, client and session ids). An absent optional
means not measured and never 0, which matters most for `thinking_tokens`: serve always
writes it, 0 included, while `generate` and `chat` count thinking only under a think
budget. Seven
surfaces name themselves: `generate`, `chat`, `batch`, and `serve:{anthropic,openai,
native,batch}` — a served batch is submitted on the native dialect but costs nothing
like a native generation, so it gets its own label rather than being folded in.
Recording is on by default; `XWEN_METRICS_FILE` points it elsewhere or turns it off with
`off` in any casing, an empty value counting as unset. A failing write prints one
warning for the life of the process, which matters for a server that would otherwise
repeat itself once per request, and never fails the run. The checkpoint name is the
official full name when the GGUF identifies as one and the file's own stem when it
identifies as nothing, spelled identically on the one-shot surfaces and on the wire, so
a custom GGUF cannot appear under two names depending on which surface ran it.

**The three choices with a why.** The file is under the XDG state directory rather than
`~/.cache/xwen`, because a lost cache costs a re-download and a lost history is gone —
and `rm -rf ~/.cache/xwen` is a thing this project does routinely. A whole record goes
out in one `write_all` to an `O_APPEND` handle, so a server and a `generate` running at
the same moment interleave records and never fragments of one; the natural spelling, a
`BufWriter` or a `writeln!` pair, is the wrong one. And no dependency was added: dates
are Hinnant's `civil_from_days` and its inverse (forty lines, exact over the range), and
the local UTC offset comes from the `date +%z` shell-out the serve TUI was already
doing, moved into `src/metrics.rs` rather than written twice. Bucketing by UTC would
have removed the shell-out and cut the evening off the day the operator spent it in.

**`xwen stats` groups by `day|week|month|model|surface|client|session|all`**, filtered by
`--since 24h|7d|4w|YYYY-MM-DD` (a date means local midnight), exact `--model` and
`--surface`, substring `--client` and `--session`, with `--json` for the rows and
`--file` to read another history without ever recording into it. Every rate is
sum(tokens)/sum(seconds) over the bucket, never a mean of per-run rates: averaging rates
weights a two-token reply like an hour of decoding, and summing both sides first is also
the only version where the total row and the rows above it are computed the same way.
Where a bucket measured no seconds the cell is `-`, because zero tok/s is a claim and no
measurement is not. Malformed lines are skipped and counted in the stderr footer
alongside the path and the run count, so a piped table is nothing but its own rows. The
footer also says when `date +%z` could not be read and the days were bucketed in UTC,
that report being otherwise indistinguishable from one on a machine that really is UTC.

**`cached` needed a per-surface decision and batch needed two independent counts.** Serve
is the only surface with prefix reuse, so its cached count is the job record's
`cache_read`. `generate` runs from a reset cache and `chat` re-prefills the whole
conversation every turn, so both record 0 — that is the chat surface working as
designed, not a gap. Batch is the one that cannot be summed straight off the items: one
record covers the whole run, and every item's `cached_prefix_tokens` counts the shared
prefix, so adding that column multiplies the prefix by the item count. Cached is that
sum minus the shared prefix, prefilled once and read back for every item after. Prefill
is measured on its own as what the engine forwarded, the runner's figure plus the shared
prefix, whose tokens and time sit outside the runner's accounting in
`shared_prefix_tokens`/`snapshot_ms` — hence the recorded seconds being `prefill_ms +
snapshot_ms`. Neither count is derived from the other, and `prompt = cached + prefill`
is an ordinary-run identity rather than an invariant: a scored batch forwards more than
its prompt, an abandoned or failed serve job less. The `/xwen/v1/batch` route's
`BatchSummary` folds both prefill halves the same way.

**A failed run is a record, not a gap, and `ok` is stricter than "no error".** A history
that only holds successes reads a bad afternoon as an idle one, so every error path
writes a record. `ok` means the run reached an end-of-generation token or its token cap
with nothing thrown: a served job the client hung up on, one a deadline killed, and a
chat turn cancelled with Ctrl-C are all `ok` false, because their counts stop wherever
they were interrupted and averaging them in drags every rate down. Those records still
carry the tokens they reached. Zero counts mean zero measurements — a `generate` whose
generation errored, a turn that died in prefill, a CLI `batch` that failed as a whole
request, a served batch holding only the queue's bytes estimate. Every surface records
its own failures, so the only run that leaves nothing behind is one whose process was
killed. A served batch takes its surface from the job's kind at pickup
rather than from a summary it may never have written, so one that dies before the runner
returns is still a `serve:batch` row and not a `serve:native` one. On the reading side
`ok` separates populations rather than filtering them: a bucket sums every record the
query matched, unfinished ones included, and reports how many of them were unfinished in
the `--json` rows, the table keeping its columns. A batch counts as failed only when
every item failed. A CLI `batch` failure records the model as `-`, since most
whole-request failures happen before the payload names a checkpoint. The
acknowledged hole is process-level Ctrl-C: `generate` installs no signal handler and an
interrupted one leaves nothing behind, where the REPL already owns Ctrl-C as a cancel
and records the turn.

**The identity research, and what it settled.** Both the session and the client id were
accepted-and-dropped before this arc, and the question was which one a report should be
keyed on. `x-claude-code-session-id` is Anthropic's documented per-session header for
gateways and Claude Code sends it on every request, so it is a single value with a
single meaning and needs no parsing. `metadata.user_id` (and OpenAI's `user`) is the
opposite: undocumented, and observed in two different shapes across Claude Code
releases — an underscore-joined `user_<hex>_account_<uuid>_session_<uuid>` in one
published capture, a JSON blob carrying `"session_id"` in another. Parsing it at write
time would bake this session's reading of an unstable format into a file meant to
outlive it, so the raw value is stored and `session_key` does the interpreting: header
first, then whatever follows the last `session_` marker, then `-`. Both ids are cut to
128 characters, the file being the one place a client's own bytes land unfiltered.
`x-claude-code-agent-id` is deliberately not recorded — it would split one session into
a row per subagent. **Not confirmed:** whether the header's id is the same id
`claude --resume` lists as the transcript id. Nothing here depends on it, but a report
that claimed it would.

**The review round.** Twenty findings came back before this shipped and all twenty are
fixed. Five were material. Batch accounting was wrong in both directions: cached was
taken as prompt minus prefill, which goes negative and saturates to 0 on a scored batch,
and prefill excluded the shared prefix whose time was nevertheless being counted, so the
run's own prefill rate was inflated. A served batch that failed before its runner
returned was filed as `serve:native` with a native generation's numbers, because the
surface was read off the presence of a summary rather than the job's kind. Reading the
client and session ids had narrowed the two request fields enough that a wrongly-typed
`user` or `metadata` became a 400, which is this server's accept-and-drop policy broken
by a feature that only wanted a label. A custom GGUF recorded under the official
checkpoint whose graph it runs, so one file could appear under two names depending on
the surface. And a test set `XWEN_METRICS_FILE` in the process environment, which is
undefined behaviour beside every other thread in the runner; the path rule is now a
function over values (`metrics_path_from`) and the test passes it arguments. The rest
were smaller: the `off` casing, the empty-value case, the reader losing a whole file to
one torn UTF-8 line, `--since 2026-02-31` being accepted, the writer's directory
creation, error paths on the CLI surfaces recording nothing at all.

The outside-model pass that followed ran on Qwen3.8-Flash-Next through `qwen-review`,
xwen reviewing xwen. Codex could not be reached for it — the login on that wrapper is
revoked — so this round had one outside family rather than the usual two. Qwen
contributed two findings, both the same thing seen from two surfaces: a serve job
abandoned when its client disconnected or its deadline expired was recorded `ok` true,
and so was a chat turn the reader cancelled with Ctrl-C. Both stop counting wherever
they were interrupted, so both would have sat in the table as completed runs quietly
pulling every rate down. `ok` now means "reached its own end", which is what the two
cases have in common and what neither of them did. The stderr note when `date +%z`
cannot be read, without which a UTC-bucketed report is indistinguishable from a report
on a UTC machine, came from the same pass.

A second round over those fixes found four more. The table measured its label column in
`char`s, so a CJK or emoji label broke the alignment of every row under it, and a raw
client id could run the table off any terminal; the column is measured in display
columns now (`unicode-width` was already a dependency, so this cost nothing new) and cut
to 48 with an ellipsis. `generate` was the last surface whose failure path recorded
nothing, which made "every surface records its runs" not quite true and made a prompt
that outgrew `--max-ctx` invisible. The REPL's record warning was going out through a
path that does not return the carriage in raw mode. And `--since` was parsed after the
empty-history early return, so a typo on a fresh machine answered "no metrics recorded
yet" instead of saying what was wrong with the argument.

**Verification, and what it is not.** 26 tests in `src/metrics.rs` (record shape,
session-key derivation across both observed client shapes, `--since` parsing including a
day its month does not have, the civil date helpers across leap boundaries, bucketing,
the sum/sum rates, malformed and torn-line skipping, the environment rule, the label
column's width arithmetic) and serve at
442 against 428 before the arc — the surface split including `serve:batch` decided from
the job kind, the failed-batch cases, and the header and body ids captured per dialect,
malformed ones included. 11 in the binary and 82 in batch. The report path was exercised
against a hand-written fixture rather than a real history. **No model was run end to end
for the smoke** — another model process held the GPU for the session, and the
operational rule here is one large model process at a time. So the assertion "a real
`generate` lands a line in the real file" is tested at the unit level and unproven at
the surface, and running one is the first item on the ledger.

**Deferred (TODO.md, "Deferred from the metrics arc").** The `[metrics]` serve.toml
table; the bench and parity scripts recording into the same file and skewing usage
stats; `x-claude-code-agent-id`; the unconfirmed header-vs-transcript id; the file
growing without rotation, `--since` being the only bound; the single UTC offset a report
applies to all of its records, which buckets a run on the wrong local day across a
daylight-saving change; the end-to-end smoke above; and the serve TUI still not showing
a job's model though `JobRecord` now carries it.

## 2026-08-31 — Serve dialects: a mid-conversation system turn demotes to a user turn, and Claude Code runs against xwen end to end

**Context.** A config-repo session pointed Claude Code at xwen's Anthropic endpoint
(the rewritten `qwen` launcher; `qwen --review` drives the headless read-only
reviewer). First real harness traffic immediately hit a wall: Claude Code injects its
token-budget reminders as `system`-role messages mid-conversation, and both wire
dialects carried them through to the renderer, whose template-fidelity check refuses a
system message past index 0.

**The fix, and why not the other one.** `push_turn` in serve/anthropic.rs and
serve/openai.rs now demotes a system turn past the head of the conversation to a user
turn, in place, merging into adjacent user text. Relaxing the validation instead was
considered and refused without needing an experiment: the official templates themselves
hard-raise `System message must be at the beginning.` in the message loop (both
vendored dialects; chat_template-qwen38.jinja:106), so a mid-stream `<|im_start|>system`
block is formatting the checkpoint's own template forbids and llama.cpp would 500 on.
The renderer stays faithful, chat.rs's refusal stays as the backstop, and the wire
dialects absorb the harness quirk — the same normalizer-adapts/renderer-never-lies split
the serve tree already runs on. Demotion is positional on purpose: an
"N tokens left" reminder means what it means at the turn it arrived in, and in-place
demotion reproduces exactly how Claude Code embeds `<system-reminder>` blocks in user
turns anyway. Leading consecutive system messages still merge into the one head system
block. One test per dialect; serve suite 420/420, chat 38/38; no model math touched, so
no parity run owed. Verified end to end: a full `qwen --review` session against a live
server, Qwen answering from actually-read files.

**Also shipped (39db173): `just install`.** Plain `cargo install --path .` ignores
Cargo.lock — a re-resolved metal/objc2 crate set produced a binary whose Metal-4 kernels
failed to compile at runtime (dense_mm.metal, mpp::tensor_ops identifiers undeclared).
The recipe pins `--locked`; the trap is in CLAUDE.md's operational hazards.

## 2026-08-30 (technique survey) — Perf landscape and technique survey (research, no code): no public Apple Silicon runtime is a peer on Flash-Next, four techniques survive the cut, and candle turns out to already implement MLX-style concurrent encoding

**Context.** Two research passes with nothing built: where xwen actually sits against
every public Apple Silicon runtime, and which of the techniques those runtimes and the
recent literature use are worth taking here. No `src/` or `scripts/` file was touched.
What came out is a ranked queue (TODO.md "Deferred from the technique survey") and three
refutations (decisions.md), one of which cancels the item that motivated the pass.

**Pass 1 — the landscape.** On Qwen3.8-Flash-Next there is no public peer. The best
published same-chip figure is llama.cpp Metal on an M5 Max at **33.0 tok/s decode / 966
prefill** (heretik.io) — and on the smaller IQ4_XS build (93.7 GB), not the 111 GB
UD-Q4_K_XL we run — against xwen's 46.7 / ~980. That comparison is NOT the claim to
make: it crosses files and sessions and favours us on both counts. **The honest headline
stays the same-file same-hour +13% decode / +24% prefill** from 2026-08-29. MLX cannot
run this model on this machine at all — its 4-bit build is ~163 GB against 128 GB of
RAM — and has no first-class `qwen4_exp` graph either way. The closest engineering
analogue is **BaseRT** (arXiv 2607.00501), 1.04-1.56x llama.cpp decode on Apple silicon,
but it caps at ≤30B and explicitly excludes linear attention, so it never meets this
architecture. Two more findings worth keeping: llama.cpp master runs the whole DeltaNet
recurrence as ONE fused dispatch (`kernels/gated_delta_net.metal`), which is parity with
our `delta_scan` — nobody is ahead of us there; and vllm-metal's advertised 83x TTFT is
continuous batching, not kernels, so it says nothing about batch 1.

**The one contested class is the 35B-A3B**, where MLX 4-bit on an M4 Max measures ~91
tok/s decode and one Qwen3.5 sweep reports 130.2, against our 114. Both are other
machines, and the ledger already carries the arm that would settle it (TODO.md,
"Deferred from the landscape research" — same-machine mlx-lm, same prompts, thermal
protocol). Until that runs, no lead is claimed there. Aggregator sites (llmcheck,
siliconscore, promptquorum) were read and **discarded as unreliable**: no context length,
power mode or build stated, and rows visibly recycled between sites. Do not cite
them, here or anywhere.

**Pass 2 — techniques, ranked.** Adopt, in order: (1) concurrent encoding with
dependency barriers — which the candle read below turns into a no-op, see there;
(2) llama.cpp's **64-node lookahead reorder** to widen the concurrent dispatch sets
(`ggml-metal-common.cpp:300-370`); (3) **BaseRT-M5's gate/up + SiLU fused onto the
cooperative-tensor accumulators**, a prefill lever for `mm_id`; (4) a **command-buffer
cadence audit** (~50 ops per command buffer today); (5) a per-SKU tuned kernel-config
table with decode as its own domain. Skip or refuted: tensor cores for the decode gemv
(Apple's own M5 numbers put the decode gain at +19-27% against +28% memory bandwidth —
it tracks bandwidth, not arithmetic; BaseRT-M5 and MLX both keep decode on SIMD
kernels), graph-split reduction (llama.cpp PR #27880 measured splits 4 → 2 on
qwen4exp/M5 at prefill 665.65 → 665.27 and decode 27.99 → 27.29 — nothing, and slightly
the wrong way), BaseRT's zero-alloc decode loop (we already have it, and their papers
ablate nothing — every headline in them is end-to-end), and sorted-gather MoE and paged
varlen attention (both batch-1 irrelevant). The three that generalize are written up in
decisions.md "Refuted perf directions" with their numbers.

**The candle finding, and it is what re-ranks the queue.** A read-only map of the pinned
rev (21cca0b) says **candle ALREADY implements the MLX scheme** item (1) was going to
adopt: `computeCommandEncoderWithDispatchType(Concurrent)` (candle-metal-kernels
`command_buffer.rs:24`), one long-lived encoder per command buffer, a dependency-tracked
`auto_barrier` whose hazard sets cover the full window since the last barrier
(`encoder.rs:104-149`), fences plus untracked buffers across encoder boundaries, and a
commit every `CANDLE_METAL_COMPUTE_PER_BUFFER` — default 50, counted per DISPATCH
despite what the doc comment says (`commands.rs:18,162`). xwen's 137 dispatch sites all
bind through `set_input_buffer`/`set_output_buffer` and participate fully; nothing here
opts out of it. The encoder breaks where you would expect: the per-step logits readback
(`sampler.rs:257`), the scoring readbacks, explicit synchronizes, blits, and the
50-dispatch rollover — roughly 77 dispatches per decode step, so about two rollovers a
token.

So the lever was never "adopt concurrent encoding"; it is that candle does it COARSELY.
Ranked residual, all of it candle-side rather than kernel-side: (1) the cadence A/B on
`CANDLE_METAL_COMPUTE_PER_BUFFER` (benching as this was written; its own entry follows);
(2) whole-scope barriers → per-resource scoping, a candle patch; (3) dependency-filtered
cross-encoder fence waits, also a candle patch — today every new encoder waits on every
live fence, which is precisely the "fence-wait pileup" the 2026-08-08 prefill-residual
entry left standing as unconfirmed; (4) the CPU-side locking per dispatch (an
`EntryState` mutex, 4-6 lock acquisitions and `HashSet` inserts per bind), the plausible
component of the fitted **8.41 µs** dispatch floor. One hazard to respect before touching
any of them: candle's pooled-buffer recycle triggers at `strong_count == 1`
with **no in-flight check** (`device.rs:488-503`), so a cadence or concurrency change
can hand a live buffer back to the pool. Every arm here is validated by the parity
gate plus greedy equivalence, never by tok/s alone.

**Verdict.** The headline claim about Flash-Next survives contact with the published
landscape and gets narrower rather than wider: same file, same hour, +13% / +24%, and no
peer to compare the rest against. The technique queue is four items long and every one of
them is unpriced, which is the honest state — the survey ranks candidates, it does
not predict wins. The one thing it did settle for free is that the most-cited idea in the
Apple-silicon literature is already running underneath us.

## 2026-08-30 (FFN glue) — the L2 fold and the shexp/hc gemms onto dense_mm: Flash-Next prefill +12% at 3803 and 7606 tokens, 35B +12%

The mm_id tile pass ended with the attribution that reset this queue: the expert gemms
are a minority of the prefill `ffn` stage. At the 2048 chunk the stage is 22
dispatches/layer (8 gemms, 14 glue); per layer the rescale chain makes six elementwise
passes over the [t,10,640] activation (~367 MB), the shared expert runs three Q8_0
QMatMuls through candle's 32-token-tile mul_mm (~334 MB), and the hc `down`/`up`
(Q8_0 [320,10240] / [10240,320]) ran plain QMatMul outside the stage entirely (~40 GB
of weight re-reads per chunk). Three levers, each behind its own switch:

1. **The L2 fold** (`ops::silu_mul_l2`, `XWEN_ACT_L2_CLASSIC` reverts): the rescale
   branch's seven dispatches — silu_mul, sqr, sum_keepdim, sqrt, clamp, affine,
   broadcast_div — as one threadgroup-per-row kernel (row in registers, fixed
   sequential-then-tree sum). Bounded, not bitwise: 3.574e-7 max-rel act_s / 2.287e-7
   col_l2 against the chain (`l2_fold_matches_candle_chain`); the strict tier is
   structurally blind to it (mv_id has no rescale branch), mm/decode/ppl grade it.
2. **shexp onto dense_mm** (`XWEN_SHEXP_QMATMUL` reverts): the three projections
   through the new `QLinear::forward_gemm` above `dense_mm_min_seq` (32) — the P8c
   kernel, its precision class measured 3.68-3.70e-4 rel_l2 vs the QMatMul route at
   the shexp shapes (`forward_gemm_matches_qmatmul_at_prefill_and_is_forward_below`).
3. **hc down/up onto dense_mm** (`XWEN_HC_GEMM_QMATMUL=down|up|both` reverts per
   arm): planes via `qlinear_with_buffer` (each ~3.3 MB, sharing the QMatMul
   allocation), routed from the fused hc read only; 3.66-3.69e-4 at both shapes.

**Bench** (3803/7606-token prompts, `XWEN_BENCH=1 generate --no-draft --raw -n 64
--stats`, two rounds with arm order reversed, 60 s idles; `pmset -g` before/mid/after:
`powermode 0`, no `lowpowermode` key; anchors p1937: 1133.7 → 1119.6 → 1148.5
prefill, ±1.3%). Cumulative arms, prefill tok/s (decode flat at 43.0-44.1 in every arm):

| arm | p3803 r1/r2 | p7606 (1 round) |
|---|---|---|
| all-classic | 872.1 / 865.0 | 766.5 |
| +fold | 918.3 / 902.3 | 771.9 |
| +fold+shexp | 918.2 / 898.2 | 767.0 |
| +fold+shexp+hc-down | 896.7 / 913.1 | 775.5 |
| +hc-up = default | 962.2 / 976.7 | 860.5 |

Attribution within-sweep: the fold +4.8%, shexp ≈0 end-to-end (its 334 MB was small
against the stage), hc-down ≈0, **hc-up +7-11%** — the k=320 shape briefed as "may not
win" is the biggest single lever; its [10240,320] output side is where QMatMul's
32-token tiling re-read the most weight bytes. Total +11.6% @3803, +12.3% @7606. The
35B (no hc): 2754.8/2746.3 → 3089.9/3081.1 prefill (+12%), decode 106.9-108.7 flat.
The free A/B there, `XWEN_MM_ID_TENSOR_HP=1` (f32 activation tiles, which have no
rescale chain at all): 2286-2314 prefill — the f32-tile gemm loses ~25% against the
f16 tensor default, so deleting the rescale work does not begin to pay for it.
Refuted (decisions.md).

**Greedy** (temp-0, 64 tokens, p3803): forks on both checkpoints, and the bisect says
near-ties rather than a guilty lever. The all-classic arm is deterministic (rerun
byte-identical, and byte-identical to the prior session's default), yet EVERY lever
alone forks Flash-Next at the same decode token ~28 ("due to bureaucracy" → "due to
warfare"), four of five arms producing one identical alternative continuation. On the
35B the fold and the shexp route each fork at byte 208 with identical outputs,
combined at byte 214. No tool reports the top-2 logit margin at a decode step
(`logits-dump` is prompt-only), so the tie can be argued but not priced — ledgered.

**Gate** (35B): ALL PASS, six graded — the first re-pass where graded numbers moved:
mm cos 0.999631 → 0.999618, ppl Δnll 0.000791 → 0.001179, both within the frozen
floors; strict bit-exact, all three levers structurally off it. Provenance schema v9
adds `act_l2` / `shexp_gemm` / `hc_gemm`, all grandfathered "classic".

**The review find that mattered** (Opus, Codex and Qwen converged on it): the hc
planes as first implemented also opened the mv_ext 2..8-token window on every hc path
— the plane predicate is `dense_mm_supported || mv_ext_supported`, hc `down`
(k 10240) passes the mv_ext one where `up` (k 320) does not, so ragged chunks and
serve resumes would have changed numerics asymmetrically, `XWEN_HC_CLASSIC` included.
The rule now: **hc planes are dense_mm-only** — `QLinear::without_mv_ext` keeps
`forward` on QMatMul at every count the gemm doesn't take, bitwise the pre-plane
behavior (`without_mv_ext_keeps_small_batch_on_qmatmul`), while `forward_gemm`
keeps the plane. The shexp planes predate this change and keep their window:
`XWEN_SHEXP_QMATMUL` restores the immediate pre-change route, mv_ext included.

## 2026-08-30 (mm_id tiles) — work-list grid and NR1 64: expert gemms +17-23% in isolation, prefill unchanged end-to-end at 3803 tokens, and the ffn stage turns out to be mostly not the gemms [CONTESTED 2026-09-05: the "mostly not the gemms" reading came off the 2.2x-inflated stage profiler; two in-situ A/Bs bracket the expert gemms at 14-43% of prefill wall, and the tile work reads −2.8% end to end — see "Ceiling diagnosis"]

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

## 2026-08-30 (serve on Flash-Next, first benchmark) — serve decodes at parity with `generate` through 32k (42-47 tok/s), a 32k conversation resumes its next turn in 0.5 s, and an edited prompt re-prefills from zero by design

Every Flash-Next figure in CLAUDE.md was `generate`'s, and TODO.md's "a qwen4exp serve
run has never been benchmarked" said not to quote them as serve figures. This is that
run: read-only, at f949b1d, nothing in `src/` touched.

**Setup.** `xwen serve --no-tui --port 8099` on its defaults — `--ctx 262144`, 2 cache
slots, 4 snapshots, disk tier off, no drafter (Flash-Next ships none). OpenAI dialect,
streaming, `stream_options.include_usage` on for the cached-token counts,
`chat_template_kwargs {enable_thinking: false}`, `max_tokens 64`. Prompts are the qsa-c
fixtures plus ~50 tokens of chat template. Load 23.2 s. `pmset -g` printed `powermode 0`
and no `lowpowermode` key (verbatim; no high-power claim either way). 20 runs, no errors.
Scripts under the session scratchpad, `serve-bench/`.

Three passes per prompt, in order: **cold** (nothing in the cache), **cached** (byte-
identical prompt resubmitted), **partial-edit** (the last user message rewritten, the
prefix ahead of it untouched). Each length ran twice (r2).

| prompt tokens | TTFT cold | TTFT cached | TTFT partial-edit | decode cold / cached / partial | cached_tokens cold / cached / partial |
| --- | --- | --- | --- | --- | --- |
| 1984 | 2305 ms | 95 ms | 2451 ms | 45.19 / 46.87 / 43.00 | 0 / 1977 / 0 |
| 1982 (r2) | 3957 ms | 97 ms | 2813 ms | 45.41 / 46.76 / 43.85 | |
| 7655 | 12336 ms | 126 ms | 10942 ms | 44.11 / 45.45 / 43.76 | 0 / 7648 / 0 |
| 7653 (r2) | 11120 ms | 137 ms | 11416 ms | 44.65 / 44.62 / 43.75 | |
| 32108 | 63949 ms | 233 ms | 64584 ms | 43.47 / 43.53 / 41.92 | 0 / 32101 / 0 |
| 32107 (r2) | 62933 ms | 215 ms | 64534 ms | 42.92 / 42.09 / 43.14 | |

Derived prefill (the bench script's own approximation: TTFT minus its render/encode
delta, first decode step included — NOT tokens over TTFT, which gives 861 for the 2k r1
cold cell and 501 for the 3957 ms r2 outlier): **~800-940 tok/s at 2k, 627-696 at 8k,
500-511 at 32k** — the same shape `generate` shows, prefill falling with length while
decode barely moves. Treat the prefill figures as approximate; `generate --stats` is the
prefill instrument.

**Thermal check.** The 1984-token anchor read 45.14 tok/s at the start and 42.53 right
after the two 32k prefills — −5.8%, over the 3% flag — and came back to 46.20 after a
90 s cooldown. That is the duty cycle, not a level shift; the 32k rows below were taken
under it and are the pessimistic end of their own spread.

**Against `generate`.** Parity at 2k and 7.6k (44-47 either way). At 32k serve reads
**4-7% lower** — under the 10% bar, so it was not chased and the
profiler was not run. One plausible contributor, untested: serve's
`--ctx 262144` logs `state 2.0GB` where `generate` at `max_ctx 8192` logs 0.2 GB, so the
recurrent-state allocation serve walks per step is 10x the one-shot path's. Ledgered.

**Multi-turn is where serve wins.** The Claude Code pattern — `[user, assistant, user]`,
the conversation grown by one exchange — resumes off the turn-boundary snapshot: 7650 →
7744 tokens cost 10467 ms cold and then **348 ms** for the follow-up (cached 7713 of
7744); 32108 → 32202 cost 67492 ms and then **489 ms** (cached 32171 of 32202). A 32k
agent conversation takes its next turn in half a second.

**Editing the last user message gets `cached_tokens: 0` and a full cold prefill, and
that is the design, not a miss.** `PrefixCache::plan` (src/serve/engine.rs:3727) takes
the longest common prefix with what the slot holds; because the edit is not at the
cache's end it cannot extend, so it asks `rewind_to` (engine.rs:3701) for a resume
point, and `rewind_to` quantizes DOWN to the nearest snapshot — the leading system block
(the anchor), a turn boundary, a fork point, or the tail a page-out took
(`plan_snapshot_stops`, engine.rs:2009). Below the shallowest of those it returns
`Cold`. The reason is state, not bookkeeping: DeltaNet and PLE state is recurrent, so it
is restorable only at positions where it was captured, and no snapshot sits inside a
message. Editing the last message of a single-message prompt therefore lands under every
snapshot and replays from zero. Mid-message snapshots would fix it and cost ~30 KiB of
image per token retained; ledgered, not taken.

**Footprint** at rest after the 32k runs: 16 GB phys, 21 GB peak, 43 GB of clean mapped
weights.

## 2026-08-30 (QSA decode, step C) — block selection moved onto the device: Flash-Next decode above the 2048 budget 33 → 44-45 tok/s at 3.8k-32k, the cliff closed

Steps A+B (below) left one named cost: each of the 12 QSA layers read its block scores
back to the host every decode step (`to_vec1` in `select_with`), ran `top_blocks` +
`expand_into` there and uploaded the rows. Each readback drains the pipeline, so the CPU
could not encode layer N+1 while the GPU ran layer N — the GPU idled 12 times per token.

**Mechanism.** `kernel_qsa_select` (`src/ops/qsa_select.metal`, `ops::qsa_select`,
`dispatch::run_qsa_select`): one threadgroup of 1024 threads per layer per step, each
thread a contiguous stripe of `ceil(nb / 1024)` scores. The threshold is found by
MSB-first radix select over a canonical integer key — four passes of a 256-bin threadgroup
histogram, thread 0 walking the bins downward to the one where the cumulative count
reaches what is still needed. Then a compaction: per-thread counts of keys above and equal
to the threshold, two threadgroup exclusive scans (simd prefix + per-simdgroup totals) that
rank the equal keys in index order and assign output slots, and a re-walk that emits every
above-threshold block and the lowest-indexed equal ones up to the quota. Stripes are
contiguous and thread-ordered, so the rows come out ascending. `keep == nb` is an
identity fill. `n_sel = keep * ratio + tail` is known on the host (`tail = (pos+1) %
ratio` at a single-token step), so the output buffer is allocated at the right size and the
step has no readback at all: the attention's gather reads the row buffer the kernel wrote.
Prefill (`n > 1`) keeps the host path — its overlay is a host-assembled mask anyway.

**The key.** Both arms rank by ONE function, `score_key` (a Rust copy in indexer.rs used
by `top_blocks`' comparator, a Metal copy in the kernel): a non-negative finite float
orders by its bit pattern, denormals included; a set sign bit or a NaN keys as 0. This
made the host comparator a true total order (it was `partial_cmp` with an `Equal`
fallback) and makes the two arms identical for every input, contract or not. The trap
on the way: the design's `as_type<uint>(max(score, 0.0f))` FAILED the tie sweep at
`nb 100 keep 50` — Metal's `max(-0.0f, 0.0f)` returned `-0.0`, whose bit pattern is
the LARGEST key, so a `-0.0` block was kept ahead of everything. The pure-bit key also
sidesteps flush-to-zero on the compare, which would have ranked a denormal equal to 0.

Kill switch `XWEN_QSA_HOST_TOPK` (presence-based; `XWEN_QSA_CLASSIC` implies it;
stripped by parity-gate.ts; a row in parity.md's qwen4exp table). Tests: the kernel against
`top_blocks` + `expand_into` bit for bit over nb ∈ {1, 5, 100, 511, 512, 513, 2000,
65536} × keep ∈ {1, nb/2, 512, nb} × tail 0..3 with heavily tied scores (exact zeros,
−0.0, denormals) and continuous ones; a tie-quota case spanning 500 stripes; NaN and
negative scores; the dispatch's bails; and the scripted-sequence test now runs THREE arms
(classic+host, cached+host, cached+device) with identical rows at every step.

**Bench** (thermal protocol; `--no-draft --raw -n 64 --stats`; before =
`XWEN_QSA_HOST_TOPK=1`, after = default; arm order alternated between rounds — r1
before-then-after, r2 after-then-before; 60 s between rounds; `pmset -g` printed
`powermode 0` at both ends; no other model process):

| prompt | before decode | after decode | before prefill | after prefill |
| --- | --- | --- | --- | --- |
| 1937 anchor (after arm) start → end | | 45.6 → 46.7 | | 960.2 → 972.3 |
| 3803 r1 / r2 | 33.1 / 33.0 | 41.1 / 44.1 | 835.7 / 834.8 | 832.0 / 837.3 |
| 7606 r1 / r2 | 33.9 / 33.3 | 44.2 / 45.0 | 718.0 / 686.5 | 695.3 / 716.5 |
| 15972 (one round; only 10 tokens decoded before EOS) | 32.0 | 41.7 | 589.2 | 596.6 |
| 32061 (one round) | 33.8 | 45.3 | 470.9 | 473.4 |

Anchor drift +2.4% (within the 3% flag). Above-budget decode now runs at 44-45 against
the 45.6-46.7 below-budget anchor at every length up to 32k: the QSA cliff is closed,
and the per-step cost no longer grows with context in the indexer. The 3803 r1 "after"
figure (41.1) is a 32-token sample (EOS) and the 16k rows are 10-token samples; the
prefill differences flip sign with arm order and are noise. Greedy (`--top-k 1`, 64
tokens) byte-identical between arms at 3803 and at 15972. The cooperative bin walk,
the prefill readback and the earlier profile's `ple` +3.2 ms above budget are the
ledgered follow-ups (TODO.md).

## 2026-08-30 (QSA decode, steps A+B) — block keys cached per complete block and the K/V row gather fused: Flash-Next decode above the 2048 budget 30.5 → 32.9 at 3.8k, 30.3 → 33.5 at 7.6k

The decode cliff at the QSA budget (TODO.md, measured earlier today: 46 tok/s below 2048
tokens, ~30 above) had three named costs, and this arc took the first two: the pooled
block keys were recomputed from EVERY raw key on every step in 12 layers, through a
`mean(1)` that candle routes to `fast_sum_f32_strided` (one two-thread threadgroup per
output — ~1.5 M per step at 4k), and the `Rows` gather ran 3 candle dispatches per
plane per layer. The third — one host readback of the scores per layer, 12 pipeline
drains per step — is step C and is still there.

**A. Block-key cache.** A complete block's key (pool → k_norm → rope at the block's
first position) depends only on its own `ratio` raw rows, which never change while they
sit below the cache length, so `IndexerCache` now carries a derived plane
`blocks: [max_ctx/ratio, 128] f32` plus `blocks_ready`. `select` builds only the
blocks `[blocks_ready, n_blocks)`, in one batch: at decode that is no key work on three
steps of four and one block on the fourth. Every `len` write goes through one private
`set_len` that clamps `blocks_ready` to `len / ratio` (truncate, rollback, reset);
`import_rows` resets it to 0 outright, because the rows below the import were replaced,
not kept. The plane is never exported — an image carries raw rows and the next `select`
rebuilds in one batch. Pooling and the per-head score sum no longer call candle's
reduce: `strided_sum` adds the narrows in the reduce's own two-thread order,
`(r0+r2)+(r1+r3)`, which is bit-identical to `mean(1)` / `sum(0)` at extent 4 (and
up to 5; refused above, where candle's simd fold orders differently).

**B. Fused gather.** `kernel_qsa_gather_{f16,f32}` (`src/ops/qsa_gather.metal`): one
threadgroup per (selected row, head), vec4 copies, strides passed in for the head-strided
cache view. Two dispatches per layer instead of six; a copy, so bitwise.

Kill switch `XWEN_QSA_CLASSIC` runs the old full recompute and the `index_select` chain;
a scripted test (`cached_block_keys_match_the_classic_recompute`: chunked prefill
crossing the budget, decode steps, a rollback, a truncate below a block boundary,
refills, export/import) holds both arms to identical selections and the plane to a
from-scratch recompute with 0 differing elements. Accounting: `indexer_bytes_per_token`
now takes the ratio and reports 640 B/token/layer (512 raw + 128 amortized block row);
`Model::kv_bytes_per_token` on Flash-Next 30720 → 32256.

**Bench** (thermal protocol: anchor arm first and last, arms interleaved classic-then-
fused per prompt, 60 s pauses between rounds; `--no-draft --raw -n 64 --stats`,
`pmset -g` printed `powermode 0` before and after):

| prompt | classic decode | fused decode | classic prefill | fused prefill |
| --- | --- | --- | --- | --- |
| 1937 (below budget) r1 / r2 | 45.9 / 45.9 | 45.7 / 45.4 | 985.8 / 997.5 | 961.3 / 960.2 |
| 3803 r1 / r2 | 30.5 / 30.4 | 32.8 / 32.9 | 841.4 / 833.1 | 837.6 / 824.7 |
| 7606 (one round) | 30.3 | 33.5 | 736.3 | 711.0 |
| anchor (1937, fused) start → end | | 45.6 → 46.2 | 988.9 → 962.4 | |

Anchor drift +1.3% on decode, within the 3% flag. Greedy text (`--top-k 1`, 3803
prompt, 64 tokens) byte-identical between arms. Caveat: every round ran classic then
fused, so the 1-3% lower prefill in the fused arm is order-confounded (the anchor's own
prefill fell 2.7% across the session with nothing changed) and is not yet a finding.
Per-step at 3.8k: 32.8 → 30.4 ms against 21.8 ms below budget, so ~8.5 ms of the cliff
remains, consistent with the 12 readback syncs step C removes.

## 2026-08-30 (prefill chunk) — the prefill chunk goes per architecture: 2048 on the MoE checkpoints (+10% Flash-Next, +8% 35B-A3B at 3.9k tokens), 512 stays on the dense 27B where 2048 is 5-6% slower

The prefill chunk had been a flat `const PREFILL_CHUNK = 512` in generate.rs and again in
serve/engine.rs since the fork, chosen for the dense 27B's attention working set. On an
MoE checkpoint the chunk is also the expert batch: Flash-Next routes top-10 of 512
experts, so a 512-token chunk gives each expert ~10 rows per `mm_id` gemm. The A/B:
`XWEN_PREFILL_CHUNK` (new, `ops::prefill_chunk_override`), `XWEN_BENCH=1`, `--no-draft
--raw --stats`, `-n 128`, the `prefill-4k.txt` fixture (3851 tokens) and its first half
(1962 tokens), arms interleaved within each round, one process at a time, first run after
the cold start discarded, `pmset -g` printing `powermode 0` (no high-power claim).

Flash-Next, three rounds, prefill tok/s (medians): at 3851 tokens **748 / 814 / 824 /
745** for chunks 512 / 1024 / 2048 / 4096; at 1962 tokens **883 / 933 / 951 / 957** (the
last two are the same single chunk there). Decode after the 4k prompt 27.4 / 27.5 /
27.9 / 27.1 — unchanged. 35B-A3B, two rounds, 3851 tokens: **2429 / 2429 at 512 → 2641 /
2627 at 2048** (+8.4%), decode 107-108 either way. Dense 27B, two rounds, 3851 tokens:
**650 / 599 at 512 → 608 / 571 at 2048** — 2048 is 6.5% and 4.7% slower within its own
round (the between-round drift is the usual duty-cycle settling; the arms were
interleaved so it cancels). Greedy output over 64 tokens on the 2k prompt is
byte-identical between 512 and 2048 on Flash-Next. Peak `phys_footprint` (polled at
1 s): Flash-Next 17.4 GB at 512 and 1024, 19.5 at 2048, 22.5 at 4096; 35B 9.4 → 11.3;
27B 41 → 44-46. No OOM.

The dense result explains the 4096 one. The sdpa mask and the attention score tile grow
with the square of the chunk; a checkpoint with no expert batch pays that and collects
nothing, and on the MoE ones it outruns the rows-per-expert gain past 2048. So the
default is per architecture — `Arch::prefill_chunk_default`: 2048 for `Moe` and
`Qwen4Exp`, 512 for `Dense` — read through `XwenModel::prefill_chunk` by every surface:
generate/chat/batch, serve (whose own constant is gone; the snapshot economics are
position-based, `SNAPSHOT_MIN_GAIN` 1024 tokens, and never assumed the chunk), and the
logits-dump ppl pass (which used to hard-code 512 to "match generate" and now reads the
same accessor). `parity-gate.ts` strips `XWEN_PREFILL_CHUNK` from the run env. The
35B's new chunk has not been re-graded by the parity gate; the last full run was at 512.
Deferred (TODO.md "prefill-chunk pass"): a 16-row `mm_id` token tile, the hc/shexp gemms
onto `dense_mm`, the f16 rescale chain at prefill, and the unattributed decode fall from
46.7 tok/s at a 530-token prompt to 27.4 at 3.9k context, seen in every arm alike.

## 2026-08-30 (later still) — the GDN mixer arc: a per-step profiler names three targets, two of them turn out to be its own brackets, and folding the beta|alpha projection buys +4.6-4.8% on Flash-Next and +8.8% on the 35B-A3B

`XWEN_GDN_PROFILE` (ae82696) is a per-step attribution of the gated-DeltaNet block: one
stderr line per forward with every DeltaNet layer folded together, each step raw AND
floor-corrected (an empty bracket is 0.7 µs but a bracket around a single dispatch is
~0.17 ms of command-buffer round trip, so the line closes both floors and prints them),
and each step's static byte count alongside so an achieved GB/s comes with it.
`XWEN_GDN_REPS` repeats a step's work inside its own bracket. Unset it costs one
`Option` check per hook and the shipped checkpoints' output is byte-identical.

Its decode line on Flash-Next, 36 GDN layers, floor-corrected, summing to 10.8-11 ms of
a ~22.7 ms token: **scan 3.79 ms (35%, 229 MB declared, ~60 GB/s) · `attn_qkv` 2.90
(27%, 1003 MB at 346 GB/s — where `attn_gate` and `ssm_out` read 523 and 537 through the
same kernel) · `gate_proj` 1.15 · `out_proj` 1.12 · `ba_proj` 1.06 (35 MB at 33 GB/s) ·
conv 0.90 (18 MB at 20 GB/s) · gnorm+zgate 0.08**, and **288 dispatches per token**. At
prefill the whole block is 0.35 ms/token, 27% of the wall.

Three of those became a unit of work: a decode-specialized scan kernel, a diagnosis of
the `attn_qkv` rate, and folding the beta|alpha gemv into the kernel it feeds. Two were
the profiler measuring itself. One shipped.

### The scan gets its own kernel, and it is a wash — the "60 GB/s" was a bracket artifact


The scan is the largest single step on that line, and a second profiled run ranks it the
same way while pricing it differently — 29.6% of the corrected mixer, ~32 GB/s of the
bytes it declares it must move against `out_proj`'s 249 in the same run. (Both runs are
quoted deliberately: the ORDER is stable, the shares and the rates are not, which is the
first hint of what the rest of this entry establishes.) That gap is what this unit set
out to close, and the diagnosis was plausible: at
seq == 1 the general `kernel_delta_scan` runs its timestep loop exactly once, so it pays
four threadgroup barriers, the q/k threadgroup staging and a two-phase `part[][]` fold
with nothing to amortize them over.

`kernel_delta_scan_decode` is that step written as its own kernel. Same decomposition —
a threadgroup owns one head and 32 of its value columns — but the columns are handed out
four at a time so every state touch is a `float4` (one load instruction covers 512
contiguous bytes of a state row instead of 128), the fold over row slices happens with a
`simd_shuffle_xor` butterfly inside a simdgroup with only the four simdgroup partials
reaching threadgroup memory, and the q/k L2 clamp-norm is computed once per dispatch
rather than once per timestep. It keeps the fp32 state, the clamp-form norm, the tiled
K-head mapping and the most-recent-first `s_out` contract (at seq == 1 there is exactly
one plane, so a rollback trail restores it unchanged).

It is a wash. Interleaved unprofiled decode on Flash-Next: 44.7 / 44.8 tok/s on the
general kernel against 44.6 / 44.7 on the decode kernel, with byte-identical 64-token
greedy text; the 35B-A3B reads 105.5 against 105.4 / 104.4, also byte-identical.

The interesting part is why. The profile line is sync-bracketed per step — the module
says so and corrects for a dispatch floor, but a step that issues one kernel and waits
still pays a GPU round trip that the real forward overlaps away. Priced the way this
repo's own benching rules demand (batched dispatches per sync, a token's worth of layers
per iteration, one cold state per layer), the scan costs 1.35-1.43 ms/token at 36 layers
of the 48-V-head geometry, not 3.79-7.19; and a candle affine moving exactly the same
state bytes with no arithmetic costs 0.98-1.02 (medians of two runs on a machine shared
with other agents, the second the quieter; the decode kernel reads 1.27-1.41 there,
0-5% better and 0.3% of a token). The new bench `delta_scan_decode_timing` carries
that floor arm deliberately, because it is the number that says when to stop: the scan's
marginal state bandwidth, differenced across the two geometries, is 525-564 GB/s, which
is what `out_proj` gets. The kernel was already bandwidth-bound. Nothing about barriers,
fold shapes or grid sizes was ever going to move it, and the 0.4 ms/token that separates
the shipped kernel from a bytes-only floor is 1.8% of a token.

The kernel is kept OPT-IN behind `XWEN_DELTA_DECODE_KERNEL=1` — the general kernel
still runs every length by default — because a wash does not pay for a second bounded
kernel on the decode path, and what it does pay for is the arm the bench prices against.
It is graded like the general one: 1e-5 against the reference AND against the
general kernel over consecutive state-carrying steps at both geometries (which is all
three checkpoints — Flash-Next's DeltaNet layers are 16 K / 48 V at 128, the 27B's
geometry exactly), plus a rollback test that walks a verify span one token at a time and
lands where the reference scan lands at every accepted prefix. Two things this did NOT
do, both ledgered: the in-place state update (the traffic is identical either way and
the aliasing promise is not the kernel's to make), and correcting the profiler's decode
brackets, which will keep pointing at this step until they are.

### `attn_qkv` was never slow — at DRAM it is the FASTEST of the three projections

The second target looked like the cleanest thing on the line. `kernel_mul_mv_q8_0_f32_attn`
reads `attn_gate` (`[2560 → 6144]`) at 523 GB/s and `ssm_out` (`[6144 → 2560]`) at 537,
and `attn_qkv` (`[2560 → 10240]`) at 346 — same kernel, same input width as `attn_gate`,
a third off the rate. That reads like an output width the grid or the cache falls off at,
and the plane is the single largest weight read in the block (1003 MB/token across 36
layers), so a 1.5x there would have been worth ~1 ms.

`q8_gemv_shape_sweep` (src/ops/q8.rs, `#[ignore]`d) prices it instead of inferring it:
eight shapes walking both axes (2560 → 2560 / 6144 / 8192 / 10240 / 12288, and 2560 out
of 2560 / 6144 / 10240, plus 6144 → 6144), three arms each, warmup 3 and 41 timed rounds.
`reuse` re-reads one plane 32 times per sync, so whatever fits the system cache is served
from it — **not a DRAM rate and not quotable as one**. `rotate` walks distinct planes
covering ~512 MB, which is the situation a real decode step is in: every layer's
projection is a different weight and the whole model streams once per token. `synced` is
the third arm and exists only because of this diagnosis — one dispatch per flush, nothing
before or after it to overlap with, the condition `XWEN_GDN_PROFILE` measures every step
in.

The rotate arm, medians of three runs, each run a median of 41 rounds:

| plane | MB | rotate µs | rotate GB/s | reuse GB/s (cache) |
|---|---|---|---|---|
| `[2560 → 2560]` | 6.96 | 30.8 | 226 | 213 |
| `[2560 → 6144]` `attn_gate` | 16.71 | 36.1 | **464** | 712 |
| `[2560 → 8192]` | 22.28 | 45.2 | 493 | 785 |
| `[2560 → 10240]` `attn_qkv` | 27.85 | 54.6 | **510** | 823 |
| `[2560 → 12288]` | 33.42 | 63.7 | 525 | 857 |
| `[6144 → 2560]` `ssm_out` | 16.71 | 35.9 | **465** | 704 |
| `[10240 → 2560]` | 27.85 | 55.1 | 505 | 822 |
| `[6144 → 6144]` | 40.11 | 75.6 | 531 | 865 |

Four K=2560 shapes fit **t = 8.41 µs + bytes / 604 GB/s** by least squares, R² 0.99996,
max residual 0.09 µs; the three off-axis shapes were held out of the fit and land within
0.8 µs of it. The rate rises monotonically with bytes moved and there is no cliff at any
width — because the only thing varying is how much traffic the fixed 8.41 µs amortizes
over. **`attn_qkv` streams at 510 GB/s**, and against the two planes the profiler ranked
above it, it is the FASTEST of the three: 510 against `attn_gate`'s 464 and `ssm_out`'s
465, precisely because it moves 1.67x the bytes per dispatch. **The profiler's ordering
of these three is inverted relative to the kernel's.**

Two things this measurement does NOT establish, both worth stating so nobody quotes past
them. It does not reproduce 346. The `synced` arm reads 122 GB/s for `attn_qkv` because
its raw number still carries the full ~157 µs sync floor that `gdn_profile` subtracts;
reconstructing what the profiler would print — solve the floor from the reported
`attn_gate` figure (160.3 µs), subtract it from `attn_qkv`'s 227.8 — gives 67.5 µs = 413
GB/s against production's 346. Right mechanism, right direction, right order of
magnitude, wrong number. The claim the evidence supports is that **the kernel cannot
produce a 346/523/537 spread and the measurement condition demonstrably can**, not that
the artifact was reproduced. (The synced arm has its own fit, `t = 157 µs + bytes / 476
GB/s`, so that condition also costs ~21% of the marginal rate on top of the floor.) And
the fit's 604 GB/s is a marginal slope differenced between two arms of the same bench,
not an appeal to a peak-bandwidth figure this machine has never had measured — the
distinction CLAUDE.md's benching rules require.

One shape is off the line and is carved out rather than smoothed: `[2560 → 2560]`
measures 30.8 µs against a predicted 19.9, 55% over, and it is also the only cell that
moves run to run (30.8 / 36.9 / 20.2 across the three runs where every other cell held to
±0.3 µs). 1280 threadgroups is too small a grid to fill 40 cores. It was excluded from
the fit for that reason, and no conclusion here rests on it.

Machine conditions, stated because they are not ideal: `pgrep` was clean at the START of
the session and not re-verified before each run, and at least one other agent (a
qwen-review) was on this machine during some part of it — the unstable `[2560 → 2560]`
cell is visible contention. The four cells the conclusion rests on held to ±0.3 µs across
three runs, so contention did not touch them, but this was not a quiet machine. Power
mode was read only AFTER the runs (`pmset -g` → `powermode 0`), which does not establish
what was in force during them, and no high-power claim is made either way.

A geometry retune was checked at the same time and there is none available: six
`(NR0, NSG)` configurations A/B'd by temporarily editing q8.metal and dispatch.rs (both
reverted), all six passing `q8_decode_production_shapes`. The shipped `(2, 4)` is the
best at all three production shapes (36.1 / 54.2 / 36.1 µs); `(4, 4)`, `(8, 4)` and
`(2, 2)` are clean and all worse, and `(4, 2)` and `(4, 8)` are much worse but came out
non-monotonic in `n_out`, so they count as "clearly not a win" rather than as figures.

### The one thing that shipped: the beta|alpha projection folds into its own head (0261e17)

`ba_proj` is the smallest projection in the block — a `[n, hidden] × [hidden, 2·H_v]` f32
gemv over a ~1 MB weight, one candle dispatch at ~30 µs per layer — and the profiler
ranked it fifth of eight. Under the dispatch lens it is the best target in the block,
because it is a whole dispatch buying a 96-wide vector.

`kernel_delta_ba_fused` (src/ops/delta.metal) reads `x_normed` and the weight and writes
`beta` and the log-decay directly; the projection output the two-dispatch arm
materializes never reaches memory at all. One threadgroup owns `DELTA_BA_COLS` = 8 output
columns, which is fitted rather than picked: at the Flash-Next geometry 8 gives 12
threadgroups and 7.4 µs, 16 gives 6 and 9.9 µs, and 4 gives 24 and ties at decode but
loses by half at seq 8-32 (`delta_ba_timing`). Narrower than 8 and a threadgroup's run
along a weight row stops filling a cache line on its own. Each column's dot product is
split across `DELTA_BA_ROWS` = 128 row chunks and folded in a tree; a `_t4`
specialization tiles `DELTA_BA_TOKS` = 4 tokens into one threadgroup so a short verify
chunk reads the weight once per tile instead of once per token.

**It pays on both DeltaNet checkpoints, and pays MORE on the faster one** (530-token
prompt, 128 decoded, unprofiled, interleaved; the Flash-Next figure measured twice, four
rounds at the commit and again in a later session with repeats inside 0.2%):

| checkpoint | decode, classic | decode, fused | prefill |
|---|---|---|---|
| Qwen3.8-Flash-Next (36 GDN layers) | 44.4-44.5 tok/s | **46.5-46.7** (+4.6-4.8%) | 796-798, unchanged |
| Qwen3.6-35B-A3B (30 GDN layers) | 105.1 | **114.4** (+8.8%) | 2248-2268, unchanged |

The mechanism is 36 (or 30) dispatches fewer per decoded token and nothing else, which is
also why the 35B gains nearly twice as much: the saving is a fixed ~0.7-0.8 ms of launch
and gemv, and the 35B's token is 9.5 ms where Flash-Next's is 22. A fixed saving is a
bigger fraction of a shorter token. Prefill is untouched on both because a prefill chunk
takes the gemv either way.

Correctness at 0261e17, and the text claim needs stating precisely. **Parity gates: 35B-A3B
ALL PASS (six graded, summaries byte-identical to fd46c7a), Qwen3.6-27B ALL PASS (five),
Qwen3.8-27B ALL PASS (five).** Forced replay against llama.cpp on Flash-Next over the
three U7 prompts: **185/192 steps agree, 7 near-ties (margins 0.0002-0.288 logit, all
rank 2-3), 0 hard mismatches, 0 nonfinite** — against 186/192 with 6 ties at fd46c7a, the
one extra flip being a 0.0002-logit tie, which is noise at the resolution the grade has.
And the greedy text: **byte-identical to the classic arm over the graded 64-token window,
but a 128-token free run forks at about step 124 on the 530-token prompt** — the fold is
bounded, not bitwise, so a fork eventually happens and this one happens outside the window
that was graded. Then graded to step 128 on that prompt against a fresh 128-step
llama.cpp oracle: BOTH arms 124/128 with the same four near-ties and zero hard mismatches,
the same top-1 at every one of the 128 forced steps, their top-5 logits drifting apart from
0.0007 at step 10 to 0.32 by step 127 — reassociation compounding through the recurrent
state. The free-run fork sits on a trajectory that left the oracle at the step-2 near-tie,
so it has no forced counterpart; it is that drift crossing a decision boundary.

Two shape decisions are worth keeping in view. The ceiling is `DELTA_BA_MAX_SEQ` = 32 and
is deliberately SHORT of the measured crossover: the fused kernel reads the whole weight
once per token tile where candle's gemm reads it once per chunk, so the advantage decays
with n — at seq 32 the fused arm is still winning, 18.8 µs against the chain's 71.7 — but
prefill chunks are hundreds of tokens, where the once-per-tile read has never been
measured and the gemm's reuse is the reason prefill is shaped the way it is. 32 covers
decode (1) and a DFlash verify block (16) and stops there; prefill takes the gemv
unchanged, and the block closes one `ba_proj` profiler step carrying both steps' bytes on
the fused arm and both steps separately on the gemv arm, so the two stay readable against
each other. And the epilogue — the beta sigmoid, the softplus decay against the pre-baked
`ssm_a` and the dt offset — is two `static inline` helpers (`delta_ba_beta`,
`delta_ba_logdecay`) called from both kernels, so the plain `kernel_delta_ba` stays
BITWISE against the reference and the two arms cannot drift apart. The fused arm
reassociates the dot product against candle's gemv, so it is bounded and not bitwise:
2e-6 across every shipped geometry at seq 1/3/4/5/16/32, the widest (hidden 5120)
measuring 1.05e-6 on the decay. `XWEN_DELTA_BA_CLASSIC=1` restores the two-dispatch
chain; `parity-gate.ts` strips it from the run env with the other kernel switches
(parity.md).

### What the arc actually taught

Two of three targets were the instrument, and the one that paid was not on the ranking's
podium. `XWEN_GDN_PROFILE` brackets every step with a device sync, so a step that issues
one kernel pays a full GPU round trip that the real forward overlaps away; the module
corrects for a dispatch floor, but that floor is one global number while the inflation is
per step and roughly inverse to the step's byte count. **The line RANKS steps; it does not
PRICE them.** Its raw mixer total, 78 ms, is more than three whole unprofiled tokens, so
the shares it prints are shares of an inflated denominator. Price a step with an amortized
bench (batched dispatches per sync, outputs held alive) or with end-to-end tok/s, never
with a figure off that line — the same rule CLAUDE.md already carries for
`XWEN_STACK_PROFILE`'s decode stages.

What the two refutations agree on is where the lever actually is. The q8 sweep's fit puts
**8.41 µs of fixed cost on every dispatch regardless of its size** (least-squares
intercept, R² 0.99996), and the GDN block issues **288 dispatches per decoded token** — on
the order of 2.4 ms of a ~22 ms token spent launching work rather than doing it. Everything else in the block is already close
to its floor: the scan is bandwidth-bound and 0.4 ms from a bytes-only copy, the
projections stream at 464-531 GB/s at DRAM, and the only change that moved end-to-end
moved because a dispatch stopped existing. So the remaining GDN work is dispatch-count fusion
and nothing else, ledgered in TODO.md item (14): conv+silu+state into the scan (−36),
gnorm+zgate into `out_proj`'s prologue (−36), the three Q8_0 projections as one
multi-plane launch (−72). At 8.41 µs × 36 layers those are ~0.3 ms and ~+1.5% apiece on
Flash-Next, and that is the honest range — the ba fold beat it because the dispatch it
displaced was ALSO doing real work badly (a candle f32 gemv at 33 GB/s), which is not
true of the kernels those three would displace. Read the same arithmetic on the 35B-A3B
and the percentages roughly double, for the reason the fold already demonstrated there:
the same fixed saving against a 9.5 ms token instead of a 22 ms one.

## 2026-08-30 (P4, later the same day) — Flash-Next serves: the QSA indexer rows ride in `HostFullKv`, the PLE state rides on its own layer's snapshot entry, and the container goes to v4

P4 was the item that kept this checkpoint off two surfaces, and it was always one
question: what does a cache image have to carry. The answer turned out to be two
answers, because qwen4exp carries two pieces of state that no image carried before and
they are not the same kind of thing.

**The QSA indexers' raw keys are position-indexed, so a snapshot stores nothing for
them.** Every token writes its own row into `IndexerCache` — one MQA key head,
`[max_ctx, indexer_head_dim]` f32 per full-attention layer — exactly the way a
full-attention layer writes its K/V. Nothing about a row depends on what came before it,
so a rewind is `IndexerCache::truncate(pos)` and it is exact, not approximate, and a
`CacheSnapshot` needs no indexer data at all. Only the page-out path has to move bytes,
and there the rows belong beside the K/V planes they mirror: `HostFullKv` gained a `qsa`
plane set and a `qsa_head_dim`, carried through `range`, `concat` and a new `qsa_prefix`
accessor. One head is what makes that cheap — a position range of a QSA plane is a slice
of the buffer, where the K/V planes need the per-head gather they have always needed.
`export_full_kv_from`, `import_full_kv_into` and `check_full_kv_importable` all take the
indexer caches now, and the import sets `IndexerCache::len` to the number of rows it
imported, which is what keeps the `cache.len == pos` invariant that QSA selection reads
before it scores anything.

**The PLE state has no inverse, so it travels as data.** The dilated conv window
(`[hc_count*hidden, (k-1)*ngram_size]` f32) and the rolling n-gram token history (at most
`ngram_size - 1` raw ids) are summaries of everything that came before; no position
determines them and nothing reconstructs them short of re-running the prefix. So they get
an image: `PleImage { history, conv, state_len }` and `PleShape { conv_len, state_len,
history_len }` in `src/qwen4exp/ple.rs`, with `PleState::image()`, `shape()`,
`accepts(PleShape)` and `restore(&PleImage)`. `restore` clears the per-token trail and
disarms the checkpoint, the same thing `LayerCache::restore` does to a DeltaNet layer —
an image is a new history, and a rollback trail from the old one is worse than no trail.

**The layer alignment is the decision worth keeping.** The snapshot's `layers` vector
stays ONE ENTRY PER TRUNK LAYER. The PLE image rides on its layer's own entry through a
wrapper variant — `LayerSnapshot::Ple { inner: Box<LayerSnapshot>, ple: PleImage }`, host
mirror `HostLayerSnapshot::Ple`, disk tag `LAYER_PLE = 3` — rather than becoming a fourth
kind beside `Full`/`Swa`/`Linear`. The reason is that the PLE layer is ALSO a DeltaNet
layer: a flat `Ple` variant standing where a `Linear` one stands would have quietly
dropped that layer's conv and delta state, and the failure mode of dropping recurrent
state is not an error, it is a conversation that restores, runs, and answers differently.
The wrapper keeps the inner snapshot exactly what it was. Nesting is one deep and a `Ple`
inside a `Ple` is refused on both the assembly path and the read path, because a wrapper
that can wrap itself is a framing that can describe a state the model does not have.
`LayerCache::restore` and `check_restorable` unwrap to `inner` and never learn what a PLE
is — the state does not live in a `LayerCache` at all, it lives in `Qwen4ExpParts`, and
`XwenModel` is the one place that pairs the two. That half of D15 stands; only its
refusal was P2 scope.

**The container went 3 → 4, and only one of the two halves forced it.** `disk_cache.rs`
carries an invariant at the constant: the version discriminates FRAMING, never content,
because the checkpoint binding and the per-layer kind tags already discriminate content
between them. The two halves landed on opposite sides of it. The PLE state is a new
per-layer tag inside unchanged framing, so an old reader refuses that layer by its tag
and no bump is needed — the same way the DeltaNet recurrent state landed at v2. The QSA
planes sit INSIDE the existing full-attention record, after its K/V planes, where nothing
tags them: a v3 reader would parse the K/V planes, stop, and then fail on framed bytes it
never consumed, which is a corruption error over a file that is not corrupt. The bump
turns that into what it actually is — a v4 file on an older build is a clean `Binding`
rejection, the scan deletes it, and the conversation costs a re-prefill.

**What that let us delete.** `XwenModel::refuse_state_transfer` is gone, and all five of
its call sites (`take_cache_snapshot`, `restore_cache_snapshot`, `export_full_kv`,
`check_importable`, `import_full_kv`) do the real work. `Model::unservable_reason`,
`unservable_message` and `unbatchable_message` are gone with it, along with serve's
startup refusal and its fallback notice in `src/bin/xwen/main.rs` and batch's refusal in
`src/batch.rs` and `main.rs`. `Model::default_servable()` now returns `Model::default()`;
its fallback branch is dead and deliberately kept, as is `Model::servable()` itself —
that predicate is the question the cache-moving surfaces actually ask, and the next
architecture that arrives half-ported needs somewhere to say no. Its doc now says what it
answers today: true for every registry checkpoint. `xwen serve` and `xwen batch` both run
Flash-Next with no flags, `/v1/models` lists it and a request may select it by name.

Two gates did NOT move, and the ledger keeps them. `auto_fetch()` is still false, which
is now the ONLY thing gating this checkpoint on the wire: it is listed and selectable
exactly when its shards are really in the HF cache, and a request for an uncached one is
a 400 pointing at `xwen fetch` rather than a 111 GB download started by a stranger.
`supports_drafting()` is still false — D6's verify seam, no MTP or other drafter wired —
so `DraftMode` resolution logs "no drafter available" for this checkpoint and leaves
every other checkpoint's drafting alone. That path already existed and needed no change.
`Model::snapshot_bytes()` already counted the PLE conv window and
`Model::kv_bytes_per_token()` already counted the indexer's 512 B/token/layer; both were
checked rather than assumed, both are unchanged, and both are now load-bearing for
page-out sizing instead of forward-looking.

Two things went differently from what TODO.md predicted. The n-gram history was ledgered
as sequence-level state to be stored beside `CacheSnapshot::pos`, needing its own plane
type and validator because it is u32 in an all-f32 plane world. It is neither: it rides
on its layer's entry with the conv window, as raw ids inside `PleImage`, and never
becomes a framed plane. And the same bullet expected new disk `LAYER_*` tags in the
plural — there is exactly one, because the QSA rows turned out to belong to the
full-attention record rather than to a layer kind of their own.

**What the tests pin.** Six new unit tests in `src/kv_cache.rs`: a PLE image through the
whole snapshot chain, the nested-PLE refusal, PLE shapes and lengths that lie, QSA planes
through `range`/`concat`/framing, QSA shape rejections, and a qwen35 image carrying no
QSA planes whose record is byte-for-byte the old one plus two zero counts. Three in
`src/qwen4exp/stack.rs` against the tiny Mixed and Exact fixtures: snapshot and rewind
reproduce the continuation; export/import pages a conversation back in across a
displacing conversation; and an image from another geometry is refused by whichever half
disagrees — QSA head dim on the rows path, PLE conv width on the snapshot path. One in
`src/serve/disk_cache.rs` round-trips a qwen4exp segment with its indexer planes and PLE
state, plus a v3-container rejection case.

Worth being exact about what that does NOT cover: there is no serve-engine harness that
runs a real model. `page_out_live` and `page_in` are private free functions over a
private `EngineState`, and the engine's own tests use stand-in payloads. So the
equivalence is pinned one level down, at `XwenModel::export_full_kv` +
`take_cache_snapshot().to_host()` → `check_importable` → `import_full_kv` →
`restore_cache_snapshot`, which is exactly the sequence those two functions perform. The
real file through a real server — load, converse, page out, evict, page back in, continue
— has not been run and is ledgered.

Still open for this checkpoint (TODO.md, "Deferred from the qwen4exp cache-image arc"):
`IndexerCache` still allocates at `max_ctx` up front with no growth path where the
trunk's KV grows lazily, 4 MB per QSA layer at 8k, and page-in now gives that a second
edge — a conversation longer than the live allocation is refused rather than grown, since
the import can only set `len` within the plane it was handed. No drafter. The disk tier's
stored-image path is unit-tested only. And no qwen4exp serve run has ever been
benchmarked: every perf number for this checkpoint comes from `generate`, and the
one-shot figures must not be quoted as serve figures.

Two review passes over the finished diff, one of them a different model family, and
between them they moved the same check twice — which is worth recording, because the
second correction was to the first one.

The check is `PleState::accepts`, and the state it guards is the one with no inverse.
It originally bounded the restored history with `<=` the n-gram order. But the reachable
lengths are not a range: a state that has stepped no token holds NOTHING, and every state
that has stepped at least one holds EXACTLY `ngram_size - 1` ids, because `next_history`
left-pads with eos and then takes that many. An image declaring one id on a 3-gram layer
would have installed and made the hash pad an extra eos in front of the real predecessor.
So the bound became the set `{0, ngram_size - 1}`. The second pass then pointed out that
this is still wrong in the other direction: `0` is only legitimate AT POSITION ZERO, and
an empty history on a segment restoring to position 900 makes the hash pad two eos
instead of one — the same bug wearing a "fresh state" costume, and one the set-membership
form waves straight through. The rule is position-dependent and cannot be stated without
`pos`, so `pos` is now threaded from `restore_cache_snapshot` and `check_importable` down
through `check_ple_restorable` into `accepts`, and the check is a single equality against
what that position implies. The test pins both directions and was confirmed to fail
against the set-membership form.

The trigger for all of these is a crafted or rotted v4 segment rather than an in-process
snapshot — a live state at a nonzero position always holds a full history — so this is
untrusted-input hardening of a path whose own comments say it is written to be loud.

Two smaller ones from the same passes. `export_full_kv_from` now checks every indexer's
head dim instead of taking the first layer's and trusting it: at `pos == 0` every plane
is zero bytes, so `HostFullKv::new`'s per-plane length check says nothing, and a
heterogeneous stack would have described itself with one layer's geometry. And
`restore_cache_snapshot` now runs a real immutable preflight over the trunk layers. Its
comment claimed "asked of both halves before either is written", but only the PLE half
had a preflight; `LayerCache::restore` re-checks each layer inline, which refuses the bad
layer without unwinding the ones already overwritten — a kind mismatch at layer five
leaves four holding the previous conversation at lengths that still agree. Narrowly
reachable (the disk path preflights through `HostSnapshot::check_restorable`, and an
in-process snapshot comes from the same model), but a comment that claims an invariant
should be backed by code that enforces it.

1009 lib tests and 5 bin tests pass.

## 2026-08-30 (same day, follow-up) — `xwen batch` was never able to run the new default; it now shares serve's fallback, and a `--model` file identifies itself on the one-shot path

Review of the default move found two holes in it.

**`xwen batch` cannot run qwen4exp, and the default move pointed it straight at it.**
The entry above lists `batch` among the modes that "can run it". They cannot: batch
prefills the items' shared prefix once and takes a cache snapshot there
(`batch.rs:515`), and an enum-scored field snapshots and restores around every option it
scores (`batch.rs:1518`/`:1527`). Both are `XwenModel::refuse_state_transfer` on
qwen4exp, for the same reason serve is refused. So a zero-flag `xwen batch` would have
downloaded 111 GB, loaded it, prefilled, and failed at the first snapshot with a message
about cache images. `XWEN_BATCH_NO_CACHE` is not an escape hatch either: it skips the
shared prefix and leaves the per-option snapshots, so it works right up until a schema
has an enum in it, which is worse than not working.

`BatchRequest::model()` now resolves an absent `"model"` to `Model::default_servable()`,
the same fallback serve uses, and refuses `Qwen3.8-Flash-Next` by name before the load.
The CLI prints the fallback line on stderr when the payload named nothing, mirroring
serve's. A hub test asserts the two defaults are the same checkpoint, so P4 cannot make
one servable and leave the other behind.

That made `Model::servable()` a predicate about two surfaces rather than one, and its doc
said the opposite in as many words ("the CLI one-shots (`generate`, `chat`, `batch`) run
the checkpoint fine"). Corrected, along with `hub.rs`'s `servable` doc and
`batch.rs:210`. `unservable_reason()` lost its trailing serve clause and is now the model
half only, so both refusals can quote it and add what they do with a cache image;
`unbatchable_message()` is batch's. The serve line quoted in the entry below therefore
reads slightly differently now. Neither message offers the other refused surface as the
way out, which the single message did (it sent an operator to `xwen batch`), and a test
pins that.

**A custom GGUF on `generate`/`chat`/`batch` was silently taking the default family.**
With `--model <gguf>` and no `--model-size`, the one-shots used `select.size()` for the
chat dialect, the drafter and the label, which after the default move meant Flash-Next
for any file at all. Serve has read this off the file since it started identifying
checkpoints. The rule moved into `XwenConfig::identify` (returning `Identity::Official`
or `Identity::Assumed`), `serve::engine::identify_checkpoint` became a thin mapping onto
`Target`, and `one_shot_checkpoint` in main.rs is the CLI's caller. `--model-size` stays
a cross-check that has to agree; a file that identifies as nothing still falls back to
`Arch::model()` with a line. Batch reads the payload's `"model"` as that cross-check,
since it has no size flag. The metadata open happens before the template knobs resolve
and before `resolve_model` can download anything, so a contradicting flag fails in
milliseconds.

Two smaller ones. `scripts/hf.ts`'s `officialModel` checked shard 1 only, so an
interrupted 111 GB fetch resolved as a cache hit and failed deep in the load; it now
requires every entry in `shards` and names the missing ones, which is what that key was
added for (TODO.md called it dead). And the README's model section still said "all
Q4_K_M" (Flash-Next is UD-Q4_K_XL), called the checkpoint "CLI-ONLY" (it is
`generate`/`chat` only), and claimed a status for "both checkpoints" from when there were
two.

998 lib tests and 5 bin tests pass.

## 2026-08-30 — Flash-Next becomes the default checkpoint; serve falls back to the 35B-A3B and says so

`#[default]` on `hub::Model` moved from `Qwen35BA3B` to `Qwen38FlashNext`. Every mode
that can run it now runs it with no flags: `generate`, `chat`, `batch`, `fetch`, and
`bun scripts/bench.ts`. Nothing about the checkpoint changed — it is the same
EXPERIMENTAL label, the same three P2 gates, the same 111 GB of shards.

Serve is the mode that cannot run it, so it does not. `Model::default_servable()` is the
new rule: `Model::default()` when the server can run it, otherwise `Qwen3.6-35B-A3B`. A
zero-flag `xwen serve` prints one line —

    xwen: Qwen3.8-Flash-Next cannot be served yet (the qwen4exp recurrent state (the QSA
    raw-key caches, the PLE conv window and its n-gram token history) is not carried by
    any cache image, and the server snapshots, rewinds and pages conversations out on
    its ordinary path); serving Qwen3.6-35B-A3B. Pass --model-size to choose.

— and serves the 35B-A3B, which is what it served yesterday. `--model-size flash-next`
is still a startup refusal: naming a checkpoint and quietly getting another one is the
failure `checkpoint_selectable` exists to prevent. The reason clause is now
`Model::unservable_reason()`, so the refusal and the fallback line cannot drift apart.

Two things that went differently from the plan. The fallback was specified as "the first
entry of `MODELS` that is servable" — that entry is the **27B**, not the 35B-A3B, because
`MODELS` is the order `/v1/models` prints in, not a preference order. Deriving it would
have cut every existing server's decode to about a quarter (25 tok/s against 104-107)
while looking principled, so `default_servable()` names its fallback. And the fallback
notice moved inside the "nothing named a model" branch: a config file with its own
`model` path is not falling back to anything and should not be told that it is.

`auto_fetch()` is deliberately unchanged, which leaves the one real cost: a zero-flag run
on a cold cache downloads 111 GB. That gate was always about a checkpoint arriving as a
side effect of a stranger's HTTP request, not about an operator's own zero-flag run;
`ensure_model` fetches all four shards after the same size notice every other checkpoint
prints, and resumes in place.

Drafting: the default checkpoint ships no sidecar, so the zero-flag path prints "decodes
without speculation (no drafter exists for its graph yet)" rather than the old "no
drafter available" line, which read as something having gone missing.

Scripts: `hf.ts`'s `defaultSize()` follows the binary to `flash-next`. `parity-gate.ts`
now pins `35b` explicitly — it has no llama.cpp oracle for the qwen4exp graph — and
`retune-draft.ts` was already excluded by `draftingSizes()`. `spec-equivalence.ts` keeps
its explicit three-checkpoint list.

Also removed: `hub::official_drafter()`, which had no callers (serve's `engine.rs` has
its own function of the same name).

994 lib tests and 4 bin tests pass. Two new hub tests pin the default and the fallback,
and the fallback test asserts the two must converge once the default becomes servable —
so P4 cannot land and leave serve on the older model by accident.

## 2026-08-29 (P3, later the same day) — Flash-Next prefill 239 → 796 tok/s and decode 37.8 → 45: a Q5_1 `mm_id` arm, four fused hyper-connection kernels, and a norm split across streams below 32 tokens

**Context.** P2 closed hours earlier with the graph correct and prefill 3.3-3.5x behind
llama.cpp on the identical file. P3 opened on that gap. Everything below is the
`UD-Q4_K_XL` file at a 530-token prompt, arms interleaved, `pmset -g` reporting
`powermode 0` this session — no `lowpowermode` key appeared, which is the reverse of
what this machine usually prints, and is recorded verbatim rather than normalised.
High-power mode is still not positively confirmable here, so it is not claimed.

**Measure first, and the profiler had holes.** `XWEN_STACK_PROFILE` let three
Flash-Next-only costs — the PLE layer, QSA selection and the token readback — fall into
`inter_stage_host`, so they were invisible by construction. b54046b gave each its own
bracket (the qwen35 path emits none of them) and the attribution came out unambiguous:
at 530 tokens prefill is **ffn 51.0%, hyper-connection glue 34.3%** (`hc_ffn.read` 12.1,
`hc_attn.read` 11.6, the two `hc_write`s 10.6), `mixer_delta` 8.4, `mixer_full_attn` 3.7,
`ple` 1.9, `qsa_select` 0.3, unaccounted 0. At 880 tokens the same shape with ffn at 54%.
Two things were ruled out before any kernel was written: `XWEN_CHUNK_SYNC` moved prefill
by under 0.5%, so nothing accumulates across chunks, and the first profiled run of a
fresh cache was cold — 36 s load, 136 tok/s — because `XWEN_BENCH`'s warmup does not
cover shard page-in. Discard the first run after a fresh cache; that one is a trap, not a
measurement.

**The Q5_1 `mm_id` arm (8112733) took prefill 239 → 443.** `ffn_down_exps` is Q5_1 on 43
of 48 layers by the 640-column rule, `FusedExperts::use_mm` is all-or-nothing per layer,
so one unsupported down plane sent all three expert planes of those layers through
per-token `mul_mv_id` at prefill. `block_q5_1` and `dequantize_q5_1` are copied verbatim
from the pinned llama.cpp and the tile loader was already 32-block generic through Q8_0,
so the arm is small; it is instantiated for the classic, `_hp` and `_t` families. The ffn
stage went 2887 → 1031 µs/token and the gap to llama.cpp 3.30x → 1.78x. Decode did not
move (37.7), which is expected — decode still takes candle's baked
`kernel_mul_mv_id_q5_1_f32`. `XWEN_NO_MM_ID=1` is **not** the baseline for this arm: it
forces mv on all three planes and reads 225 tok/s, below the real before-arm.

**Then the hyper-connection glue (8aeed73), and the gap closed.** The read was 20 candle
dispatches, 17 of them glue around two Q8_0 gemms, and the write made three full-carrier
passes to do one FMA — twice per layer over 48 layers. Four kernels replace the glue and
the two Q8_0 projections stay `QMatMul`, because `low_rank` is 320, not 8: those are real
gemms moving ~0.7 GB of weights per token and they belong in the library. 5+1 dispatches
per layer-pair instead of 20+2; roughly 2128 → 600 hc dispatches per forward. Measured
A/B at fd46c7a: **prefill 764.9/780.9 fused against 446.0/437.1 classic, 1.75x**, with
llama.cpp at 788-790 in the same hour — the 3.5x prefill gap from P2 is gone. Stage
detail: `attn_norm` 726 → 209 µs/token, `ffn_norm` 726 → 227, the residual writes 325 →
105, prefill wall 2105 → 1279 ms.

**But fusion cost 6% of decode, and the fix was to stop being one threadgroup.**
Fused decode read 35.8/35.8 against classic's 37.5/38.0. Opus's review named the
mechanism before the profile did: at `n == 1` the fused `hc_norm` runs ONE threadgroup
per token, so each of the 97 launches per forward walked a 10240-wide row and the 160 KiB
injection head on a single threadgroup — the exact shape a candle chain of small
dispatches beats. Below `HC_SPLIT_MAX_N` (32) the norm now runs one threadgroup per
(token, stream) and the injection dot as a second kernel over the same grid (2c8d3b3).
**Decode 37.8 → 43.1 tok/s**, and it is bit-identical: same thread count and partition
for the statistics, stream-major walk for the dot, verified as byte-identical output
against both `XWEN_HC_SPLIT_MAX_N=0` and the pre-split fused build over 128 tokens.

**The PLE cost was page faults, not driver overhead, and the 6.4 ms figure was an
artifact.** The decode profile had PLE at ~6.4 ms of a 26.5 ms step — one layer taking a
quarter of decode — with a shape (~6 ms fixed per forward, ~44 µs/token) that reads like
a mid-forward device→host sync draining the pipeline. It was not. `XWEN_PLE_PROFILE`
(fd46c7a) splits the layer into hash, gather, uploads, projections, the three readbacks,
host gate, host conv and trail: the gather is **flat per token, median ~1.1 ms with 6.5 ms
spikes and only 4.7% page-cache hits** — 16 IQ4_NL rows demand-paged out of a 28.8 GB
table, per token, essentially never reused. The fixed floor is ~0.85 ms (projections
0.33, the three readbacks 0.50). The honest total is **~2.1 ms of a ~28 ms step, not
6.4**: the profiler's own syncs inflate every decode stage it brackets, so profiled
decode numbers rank stages and must never be quoted as timings. Prefill is the opposite
regime — a warm chunk gathers 8192 rows in 2 ms, a cold first chunk takes 439 ms.

**Prefetch, on the strength of that finding (ac40526).** The row addresses depend only on
token ids, so a background thread touches one byte per distinct page for the position
about to be forwarded — hinted at sample time for decode, before layer 0 for a prefill
chunk. Advisory only and never gated on the PLE gate value, which is computed mid-forward
and would serialize the lookup. The table's byte range gets `MADV_RANDOM` so a 90-byte row
does not drag a readahead window, while the whole-file `WillNeed` stays for the weights;
the row math is single-sourced through `PleTable::row_offset` so the prefetcher and the
gather cannot drift. `XWEN_PLE_NO_PREFETCH` and `XWEN_PLE_NO_RANDOM` exist for the A/B.
Measured effect: `measured 2026-08-29 with one cold prompt per arm (the same-prompt design is invalid — greedy decode hashes every arm to the same rows, so arm k warms arm k+1): median decode gather 0.002 ms with prefetch vs 0.97-1.02 ms without, PLE total 1.05 vs 2.03 ms per token, decode 45.0 vs 43.2 tok/s, pf_dropped 0; MADV_RANDOM is neutral either way (0.002 vs 0.002 with prefetch, 0.97 vs 1.02 without) and stays on only because it is harmless and switchable`.

**Correctness, twice over.** Forced replay against llama.cpp with both the Q5_1 arm and
fused hc active: **186/192 agreeing, 0 hard mismatches, 0 non-finite**, the six
divergences all rank-2 near-ties at margins 0.009-0.30 logit. U7's pre-P3 number was
189/192 on the same instrument, and the difference is six near-ties changing sides, not a
math change. Free-run greedy text forks at token 2 on a 0.0817-logit near-tie and stays
forked, which is why free-run text comparison is not a grade on this checkpoint and
forced replay is. The shipped checkpoints were re-gated after the Metal source changes:
**35B-A3B ALL PASS** (6 graded; mm cos 0.999631, ppl Δnll 0.000791) and **27B ALL PASS**
(5 graded), both at fd46c7a. 2c8d3b3 and ac40526 move no shipped-checkpoint math — the
split path is qwen4exp-only and the prefetch is non-numeric — so they were not re-gated.

**Reviews.** Fable found no defects in d5757f1..fd46c7a; Opus found no correctness
defects and eight lower items, of which the parity-gate `baseEnv` leak was real and fixed
in 6eaf980 (the gate was inheriting `XWEN_HC_CLASSIC` and `XWEN_PLE_PROFILE` from the
caller's environment into the run env, which would have graded the wrong path silently)
and the decode single-threadgroup finding became 2c8d3b3. Second round on
6eaf980..ac40526: Fable-2 no defects, having gone and verified that candle's encoder
inserts the barrier the split kernel pair depends on; Opus-2 no correctness defects and
six polish items; Qwen no high findings. A polish commit (188ba73) took the rest of both
rounds — among them `fused()` now checking the weights it binds rather than only the
carrier, so the documented fall-back to the candle chain actually holds, and a bitwise
test that had silently drifted onto the split pair being pinned back to the single
kernel. Four low items are knowingly not fixed and are ledgered rather than dropped:
`n == 0` bails inside the fused hc path, the `-0.0` caveat on the bitwise tests (they
compare raw bit patterns, so a sign-of-zero difference reads as a mismatch),
`scripts/hf.ts`'s flash-next entry widening what `--model-size` the parity gate accepts
without any fixtures behind it, and that entry's dead `shards` key.

**The 15 GB of private memory is the design, not a leak.** P2 recorded that xwen dirties
~15 GB where llama-server dirties 751 MB on the same file, and it read like a bug. A
code-reading audit accounts for **~11.4 GB of it deliberately**: attention and GDN
projections dequantized to f16 planes for the prefill gemm (~5.35 GB raw, ~6.14 after
candle's pow2 buffer rounding), `token_embd` dequantized whole to f16 (1.27 → 2.15 GB,
plus a ~2.5 GB f32 transient), non-aliased Q8_0 copies for lm_head, the hc down/up
projections, the shared expert and the PLE k/v (1.65 GB together), the transposed router
weights at 8 MiB granularity across 48 layers (0.40), indexer raw-key planes sized at
`max_ctx` (0.81 at the 131072 default), and delta state (0.15). Every one of those is a
pattern the three shipped checkpoints already run; what is aliased is aliased correctly —
the 77.5 GB expert stacks, the 28.8 GB PLE table, the BF16 indexer projections. So the
"15 GB leak" reading is refuted. Three ways to shrink it are ledgered: alias the Q8_0
planes that only ever feed `QMatMul`, grow the indexer planes on demand, and gather
`token_embd` rows from the quantized tensor instead of materializing the whole table.

**End state, against llama.cpp on the same file in the same hour.** Prefill **795.7 vs
789**, decode **43.1 vs 41.4** before the PLE prefetch and 44.5-45.8 after — 1.01x and 1.04-1.10x, from 0.30x and 0.91x this morning.
Four rounds of interleaved medians at a 530-token prompt and 128 decoded tokens;
`powermode 0`, no high-power claim. Decode is **bimodal round over round** (42 vs 44
tok/s for the split path, 34 vs 36 for the unsplit one) with no explanation yet, which is
ledgered. The P0-pause notes' ~50 tok/s decode figure was always a scaling guess from the
35B-A3B; it is retired in favour of 43.1 measured.

## 2026-08-29 — Qwen3.8-Flash-Next runs: P0 through P2 in one day, agreeing with llama.cpp at 189/192 forced-replay steps, 37.5-38.1 tok/s decode

**Context.** The qwen4exp arc had been paused since 2026-08-26 with P0 complete and no
runnable weights. Three things unblocked it at once: Unsloth published the full quant
ladder, llama.cpp merged qwen4exp support, and the machine had disk free. The day ran
P1 and P2 end to end. `docs/qwen4exp-port.md` is the arc doc; the decisions moved to
decisions.md "Qwen3.8-Flash-Next (qwen4exp)" as this entry was written.

**Picking a file.** Header-parsing every published GGUF turned up a rule nobody had
written down: `ffn_down_exps` is `[640, 2560, 512]`, and 640 % 256 = 128, so it fails
every K/IQ type's block-size requirement and llama.cpp's generic `tensor_type_fallback()`
silently demotes that plane to a 32-block type on every publisher's file. Q4_K becomes
Q5_0, Q5_K becomes Q5_1, Q6_K becomes Q8_0. `per_layer_token_embd` (ncols 160) is
32-block-only for the same reason, which is why ggml-org's Q8_0 file carries a Q4_0 PLE
table. `UD-Q4_K_XL` was chosen — 111.33 GB, 4 shards, 82.53 GB of trunk — as the only
Q4-class file whose types we already have kernels for, with the consequence that 43 of
its 48 layers carry Q5_1 experts. That looked like a blocker for half a day and was not:
`ExpertStack` types each tensor from its own GGUF info with no whitelist, nothing
compares dtypes across planes or layers, decode reaches candle's baked
`kernel_mul_mv_id_q5_1_f32`, and prefill drops the layer to per-token `mul_mv_id`.
Correct today, slower than it should be, and now a P3 ledger item rather than P2 scope.

**The oracle moved.** PR #27742 merged 2026-08-27 as `6c84c7d5d`, with a graph-splits
follow-up `6fe749801` the next day. The vendored copies under `reference/qwen4exp/` were
re-vendored at the new pin, and then deleted outright: the submodule was bumped
e9fa0781 → `6fe749801` instead, one oracle for every checkpoint. The proposed second
clone — keeping the old pin frozen so the 3.6/3.8 floors stayed valid — was refuted by
just re-running the gate: **all three existing checkpoints re-passed at the new pin the
same day**, floors unchanged and not re-derived (35B-A3B six graded checks, strict cos
1.000000, mm 0.999631, ppl Δnll 0.000791; 27B clean throughout; 3.8-27B five checks,
its ppl tier skipped because that checkpoint has never had a reference fixture — a gap
that has been shipping quietly and is now ledgered). The semantic diff of the vendored
move survives as `reference/qwen4exp/UPSTREAM-DIFF-2026-08-29.md`; the one finding in it
that changed our plans is that upstream had already moved PLE conv state onto its own
recurrent row, which refuted a note we had written about upstream wasting it across all
36 recurrent layers.

**P1: three frozen references, three reviewers.** `ref_hc`, `ref_ple` and `ref_qsa` —
hyper-connections plus both norm flavours, the n-gram hash plus its injection layer, the
QSA indexer — graded against five golden fixtures, 38 tests. The review round changed
real things: the grouped norm now accumulates in f64 like ggml's CPU path and is one
shared implementation instead of three near-copies with weaker asserts, every matvec
asserts its full shape, the PLE conv dilation is derived rather than loaded, and the
gate propagates NaN. The fixtures also settled a question the port doc had recorded
wrongly: HF DOES clamp inside the signed sqrt, so a "PLE gate clamp divergence" we had
written down was retracted.

**P2: eight units, and first light.** U0 loader-owned GGUF header parsing (the real file
had never opened — `gguf::open` failed with "unknown dtype for tensor 20" until this
landed; 1223 candle tensors plus one raw IQ4_NL plane), U1 registry, U2 device
hyper-connections, U3 the QSA indexer and the attention overlay, U4 PLE and IQ4_NL row
dequant, U5 the `ZGate`/`sum_floor` wiring, U6 the stack itself. Suite 957/957. First
smoke answered "The capital of France is **Paris**." and stopped cleanly; the second,
with thinking on, produced 400 coherent tokens — reasoning inside `<think>`, a clean
`</think>`, then a working Python function with a unittest.

U3 cost a day's worth of confusion to a candle bug worth reporting upstream: **Metal
`index_select` is silently wrong on strided sources.** No error, just wrong rows. Worked
around by gathering per head.

**U7: it agrees with the oracle.** The parity gate itself cannot run on this checkpoint
at all — every tier's reference side is `--moe-impl reference`, and
`ReferenceExperts::forward` panics at `src/moe.rs:198` with "the len is 512 but the
index is 1073971200". That index is `0x40038000`, the f32 bit pattern of 2.0547, so an
f32 buffer of routing data is reaching a `to_vec1::<u32>()` read as expert ids on the
512-expert / top-10 geometry. It reproduces identically through the fused router kernel
and the candle chain, so it is downstream of the router branch. The fused runner is
unaffected, which is what made the rest of the measurement possible.

So U7 rebuilt the decode tier by hand with llama.cpp standing in for the blocked
reference runner: `llama-server /completion` fed a token-id array (bypassing both the
chat template and llama.cpp's tokenizer, so both engines provably see the same
sequence), then xwen teacher-forced along the oracle's own trajectory with `--replay`.
Tokenization was verified first and is byte-identical over the whole 4218-token corpus,
first differing index −1. Result: **189/192 agreeing, zero hard mismatches**, the three
divergences being near-ties at 0.2876, 0.0348 and 0.0097 logit — far inside the
`NEAR_TIE_MARGIN_Q8` band, and in every case llama.cpp's pick was xwen's own rank-2.
Full-vocab logprobs against the oracle's `n_probs` matched at top-1 and top-5 in exact
order, first differing at rank ≥5 among entries at logprob −12 to −14. The perplexity
comparison is protocol-limited and deliberately not called a grade: 0.5413 nats for
llama.cpp's eight independent 512-token windows against 0.3697 for xwen's one continuous
4218-token context, a gap the protocols predict. Full record in
`docs/qwen4exp-parity-2026-08-29.md`.

**Review round.** Claude, Qwen and Fable reviewed the arc; Codex was out of credits.
Three findings mattered. **PLE rollback ignores commit** — a state-machine bug in the
new recurrent path, the class of thing that only shows up after a rewind. **Serve would
500** on a qwen4exp target, because the snapshot path cannot carry the new state, so
Flash-Next is now explicitly CLI-only and `xwen serve` refuses it until P4 — the honest
version of "serve integration follows CLI bring-up". And a **name-fold collision** in
the space↔hyphen identification added for this checkpoint (the file calls itself
"Qwen3.8 Flash Next").

**Numbers, with the usual caveats.** Plain decode 37.5-38.1 tok/s against llama.cpp's
40.9-41.5 on the same file in the same hour — within ~8%, unremarkable. Prefill 203.5
tok/s against 713.4 at 530 tokens: **3.5x slower, reproduced to the centisecond across
two runs**, so not pipeline compilation. Two suspects, both already ledgered: the 43
Q5_1-down layers prefilling through the per-token fallback, and the dense-FFN prefill
gemm that was exactly this shape of problem on the 27B and took a vendored kernel to
close. `lowpowermode 0` with no high-power claim, single runs, a machine shared with
other agents' builds all day, and llama.cpp thermal-boosts harder than xwen — the ratio
is the trustworthy part. One more finding worth understanding: xwen dirties ~15 GB of
private memory where llama-server dirties 751 MB on the identical file, i.e. ~15 GB of
weights materialized rather than aliased from the mapping.

**Verdict.** The graph is correct, on the evidence of three independent instruments, and
it is honestly slow in one specific place. P2 is closed and the arc pauses here. What it
is NOT: it has no drafter, no snapshots, no serve, no parity harness of its own, and no
perplexity floor — and the frozen ppl corpus looks contaminated for this checkpoint
(PPL 1.45 on WikiText-2 test where the 3.6 pair scores 1.69 nats, with llama.cpp
independently agreeing the model is that good), so it will need a fresh corpus before it
gets one. All of that is in TODO.md with dates and context, none of it silently dropped.

## 2026-08-19 (later still, same day) — Batch stops pinning the template effort at xhigh: `reasoning_effort` lands on items and defaults, refused per item on 3.6

The dialect arc gave every surface an effort knob except one. Batch's
`resolve_render` built its `ChatOptions` from `for_dialect` and overrode only
`enable_thinking`, so a batch item that enabled thinking on the 3.8 checkpoint always
rendered the xhigh preamble with no way to ask for `low` or `medium`. Spotted
reviewing the shipped arc's coverage; closed same day.

`reasoning_effort` is now a field on both the item and the request's `defaults`,
layering exactly as `thinking` does — item over defaults over the template's own
`xhigh`. On the wire it is a JSON string in the template's three spellings:
`ReasoningEffort` gained `Serialize`/`Deserialize` through its existing
`Display`/`FromStr`, so the JSON form accepts and rejects exactly what the CLI flag
does, and the key is known to the strict `deny_unknown_fields` wire types while
typos stay refused. On a 3.6 checkpoint a supplied effort — either layer — is
refused with the checkpoint named and the 3.8-template provenance stated, matching
the CLI startup error and serve's 400; batch's refusal lands PER ITEM, the module's
shape for validation failures (a defaults-level effort fails every item identically,
since the defaults reach the renderer only through the items — decisions.md
"Serving", the effort-refusal decision's second extension). Effort with thinking off
(batch's default) or alongside injected reasoning is accepted and inert/independent,
as on the other surfaces: the 3.8 template reads the level only inside
`enable_thinking`'s guard. Note the asymmetry the tests pin: an effort-ABSENT
thinking item renders the template's xhigh sentence; `medium` is the level that
renders no preamble at all.

**Verification.** `cargo build --release` clean; `cargo test --release` fully green —
849 lib + 3 CLI + 69 parity-harness tests passed, 0 failed (6 new batch tests).
`cargo fmt --check` clean. No model math touched; the parity gate was not run and did
not need to be.

## 2026-08-19 (later, same day) — The dialect arc's review pass lands three fixes: reasoning retention moves wholly to the renderer, --no-think rejects armed think budgets, and a request-level template effort on 3.6 is a 400

The arc below went through review before closing: one internal reviewer came back
clean, the external second-model reviewer (a different model family, per the working
rule that different families catch different bugs) found all three defects fixed here.
Nothing else changed; the docs entries this pass touches are annotated in place.

**Reasoning retention is decided ONCE, in the renderer, per dialect.** Both compat
normalizers (openai.rs, anthropic.rs) were stripping assistant `reasoning` from every
turn before the trailing assistant/tool run — a rule from before the dialect arc, when
the renderer dropped exactly those turns anyway, so stripping early was invisible.
After the arc it was load-bearing and wrong: the Qwen38 dialect's
`preserve_thinking=true` default (and the OpenAI kwarg `preserve_thinking: true` on
any checkpoint) asked the renderer to keep reasoning the normalizers had already
destroyed, and the three dialects disagreed — native replayed everything while the two
compat APIs silently didn't. Now NATIVE tools mode passes reasoning through on every
assistant turn and `chat.rs`'s dialect rule (`preserve_thinking || index > last_query`)
is the single owner of what renders; the `trailing_run_start` predicates are deleted.
The debug tools modes still drop reasoning everywhere, deliberately (they render the
pre-tools history). Noted in decisions.md: Anthropic's real API strips non-trailing
thinking and this dialect deliberately does NOT emulate that — it serves Qwen
checkpoints, whose 3.8 card recommends preserved thinking for agent workloads.
End-to-end tests in all three dialect files render a normalized multi-turn history
under both dialects: 3.6 drops the pre-last-query reasoning, 3.8 keeps it, and
`preserve_thinking: true` keeps it on 3.6 too.

**CLI: `--no-think` with a nonzero `--min-think`/`--max-think` is a startup error**
(both gen and chat arms, `ThinkArgs::check_think_budgets`). The ledger item had filed
the combination as "inert"; the review found it worse — the CLI arms the ThinkBudget
machinery unconditionally, so the ceiling would have waited for a `</think>` a
no-think reply never emits and eventually forced its wrap-up sentence and a stray
`</think>` into the answer. Serve already guards the same hazard by dropping the
budget when the prompt does not start in thinking; the CLI now refuses it up front,
same pattern as the `--raw` combos. TODO.md item annotated done, same day.

**A request-level TEMPLATE effort on a 3.6 target is a 400, not a silent no-op.**
`chat_template_kwargs.reasoning_effort` (OpenAI) and `reasoning_effort` (native)
reached the renderer but only the 3.8 dialect consumes them — on a 3.6 model they
changed nothing, contradicting both the CLI's startup error and the module's own
strict-kwargs stance. Both `prepare`s now take the resolved `Target` and refuse the
request with the model named; the OpenAI error points at the top-level
`reasoning_effort` field, which keeps its budget semantics on 3.6 and stays accepted,
as do kwargs `enable_thinking`/`preserve_thinking` (real 3.6 template parameters) and
the server-wide `[thinking] effort` default (an operator setting, inert-but-legal on
3.6 as before).

**Verification.** `cargo build --release` clean; `cargo test --release` fully green —
843 lib + 3 CLI + 69 parity-harness tests passed, 0 failed (5 new serve tests, 1 new
CLI test; the four tests that pinned the old normalize-level strip are rewritten to
pin the pass-through contract). `cargo fmt --check` clean. No model math touched; the
parity gate was not run and did not need to be.

## 2026-08-19 — The chat template becomes a per-checkpoint DIALECT: 3.8 gets its reasoning_effort preamble and preserve_thinking default, gen/chat gain --no-think and --reasoning-effort, and sampling defaults go mode-keyed (1.0/0.95/20 thinking, 0.7/0.80/20 instruct)

Two commits, one arc (a2e02d0, 205d9ba). The first makes the renderer
dialect-aware; the second surfaces the knobs on every entry point and re-keys the
sampling defaults. Together they close the divergence recorded on 2026-08-14: every
default 3.8 conversation was rendering without the system instruction its template
injects, equal to the official `reasoning_effort="medium"` rendering rather than the
official default.

**`ChatDialect { Qwen36, Qwen38 }`, from `Model::chat_dialect()`.** One renderer
still serves every checkpoint — the generation prompt and every turn render
byte-identically between the templates, which is what made a parameterized renderer
honest — but `ChatOptions` now carries the dialect plus a `ReasoningEffort { Low,
Medium, Xhigh }`, and `ChatOptions::for_dialect` carries each template's own
defaults. What the 3.8 dialect renders that 3.6 does not, each pinned by a test:

- The `reasoning_effort` system preamble, verbatim template prose (the xhigh and low
  sentences are constants asserted to appear character-for-character in the 3.8
  template and NOT in the 3.6 one, lengths included). `xhigh` is the template's own
  default; `medium` injects nothing (so a medium render is byte-equal to a 3.6 render
  of the same conversation); rendered only while thinking is on. With no system
  message the dialect synthesizes a system block to carry the sentence — and that
  block anchors the prefix cache like a client's own, with the preamble kept out of
  the client-content spans. With tools, the sentence opens the block AHEAD of the
  `# Tools` header.
- `preserve_thinking` defaults TRUE (3.6 stays false): a replayed 3.8 turn keeps its
  reasoning where 3.6 drops it once superseded.
- An empty system message emits NO block, where 3.6 emits the empty block its
  template unconditionally writes.
- No inline `<think>`-in-content splitting: `split_reasoning` now runs under the 3.6
  dialect only, so a 3.8 turn that carries a `<think>` block inside content renders
  it verbatim, as its template does. (The 2026-08-14 record claimed xwen "never
  implemented that fallback" — wrong: it existed and ran unconditionally; it is now
  gated, not added.)

**`TOKENIZATION_RULES_VERSION` 2 → 3**, because the same 3.8 conversation now
encodes to a different token stream and a stale disk image must not
longest-common-prefix-match a stream the current rules would never produce.

**CLI: `--no-think` and `--reasoning-effort <low|medium|xhigh>` on gen and chat.**
`--reasoning-effort` on a 3.6 checkpoint is a startup error naming the checkpoint —
the flag would change nothing there, and this repo's flags cross-check instead of
shrugging (the `--model-size` rule); unset costs nothing to allow, since the default
level renders nothing on 3.6. Both are rejected with `--raw` (the same distortion
class as the existing guarded combos: they describe a template a raw prompt never
renders), and both validate before the 20 GB load. A repl side-fix rode along:
`stream_reply`'s `in_think` was initialized unconditionally true, which under
`--no-think` would have filed an entire reply as hidden reasoning; it now follows
`enable_thinking`, and the cancel-retry rule stops waiting for a `</think>` that a
no-think turn will never emit.

**Sampling defaults are now keyed to thinking mode, per the official cards.** All
three checkpoints' HF cards recommend the same two sets: thinking temp 1.0 / top_p
0.95 / top_k 20 (what generation_config.json carries and what everything always
used), non-thinking ("instruct") temp 0.7 / top_p 0.80 / top_k 20.
`SamplerOptions::recommended(thinking)` is the single source; `Default` stays the
thinking set, so every mode-less path (raw prompts, benches) samples as it always
has. The CLI's `SamplingArgs` became Option-valued (the DraftArgs pattern — a
mode-dependent default cannot live on the flag), and serve's
`DEFAULT_TEMPERATURE`/`TOP_K`/`TOP_P` constants are gone: `ServeSettings` sampling
keys are Options, resolved per request AFTER the request's thinking mode is known,
as request value → server-configured value → mode recommendation. A pinned server
value pins one number for both modes; unset gives each request its mode's own. The
cards also recommend penalties (presence_penalty 1.5 on some arms) and a third
"precise coding" set — neither shipped, both ledgered with the reasoning (TODO.md;
the penalties one is a real entanglement with the speculative verify path, not an
oversight).

**Serve: one `reasoning_effort` field drives BOTH the think budget and the 3.8
template preamble.** The OpenAI dialect's field keeps its budget mapping unchanged
(none=off, minimal=1024, low=4096, medium=16384, high/xhigh/max=uncapped) and now
also selects the template level, nearest-mapping the levels the template does not
define: minimal→low, high/max→xhigh. That is a deliberate divergence from llama.cpp,
which passes the raw string into the template and lets the jinja raise — serving the
nearest level beats a template error, and the nearest level is what the client
meant. New `chat_template_kwargs` request field (the official card's / vLLM's
shape) accepting `enable_thinking`, `preserve_thinking`, `reasoning_effort` — the
LAST of these takes the template's three levels only, since it is the raw template
parameter. Kwargs are validated STRICTLY: unknown key, wrong type, or an off-scale
level is a 400 naming the offender, the one exception to the dialect's
accept-and-drop permissiveness (a dropped sampling param degrades a completion; a
dropped template kwarg changes the prompt the client believes it asked for). The
top-level field beats the kwarg, llama.cpp's precedence. `[thinking] effort` in the
config and serve `--reasoning-effort` set a server-wide template-effort default
(inert on 3.6, whose template has no such parameter — which is why serving one is
not a config error). The native dialect gained `reasoning_effort` and
`preserve_thinking` fields; the Anthropic dialect's shape is unchanged (no natural
field — server-wide default applies, ledgered), but its sampling went mode-keyed
and `count_tokens` now resolves the target checkpoint and renders the preamble, so
a count matches the generation it predicts. Batch renders each item under the
loaded checkpoint's dialect (`run_batch` takes it like it takes the label); batch
thinking still defaults off, so the preamble never renders there unless asked.

**Verification.** `cargo test --release`: 838 lib + 69 binary passed, 0 failed. The
dialect behaviors are each pinned — the preamble prose against both vendored
templates, the synthesized block's cache anchor and token-boundary invariant, the
medium/thinking-off byte-equalities with 3.6, the empty-system divergence, the
inline-`<think>` divergence, the kwargs 400s, the effort-level FromStr refusing the
wider OpenAI scale, and the mode-keyed resolution order on all three dialects. No
model math changed; the parity gate was not run and did not need to be.

## 2026-08-15 (latest, same arc) — MTP Stage C: the graph is confirmed against llama.cpp by BYTE-IDENTICAL output, the sweep moves the 3.8's defaults to p_min 0.7 / depth 4 (+44-45% code, +37-38% chat over plain), and two stage-B claims do not survive being re-run

Stage C is the arc's verification half: cross-check the graph against the reference
implementation (C2), fit the shipped defaults with a real sweep instead of single runs
(C3), and write the arc down (C4). No implementation changed except the two fitted
constants and the harness edits that made them measurable.

**Machine state: `lowpowermode 0`, on AC power, 100% charged, machine otherwise idle.**
Per CLAUDE.md this machine emits no `powermode` key, so high-power mode is not claimed.
The session is calibrated by its own control: the 3.6-27B's DFlash pair read 38.0 code /
37.1 chat against 25.3 plain, against the 37.5-38.2 / 36.8-37.4 recorded on 2026-08-08 —
so this session sits exactly where the last one did, and nothing here needs the
contended-machine caveat.

**C2 — the graph is confirmed, and by a stronger result than the one asked for.** Both
implementations ran the same raw fixture, same target and sidecar, depth 3, `p_min` 0 on
both sides, greedy, 128 tokens. Acceptance: 73.3% (xwen) against 75.0% (llama.cpp) on
code, 45.7% against 47.1% on chat — 1-2 points, where 10 was the bar. But the load-bearing
result is that the generated text was **byte-identical on both fixtures**. That means
acceptance is being compared over the very same continuation rather than over two texts
that happen to resemble each other, and it independently exercises the trunk: two
unrelated implementations agreed on every greedy argmax for 128 consecutive tokens. The
residual 1-2 points is xwen proposing a few more drafts near the end of the token budget
(120 against 116, 162 against 157) — round bookkeeping, not graph.

Getting there cost two harness traps worth recording. `llama-cli` in the pinned revision
embeds llama-server and runs CONVERSATION mode regardless of `-no-cnv`: it applied the
chat template and enabled thinking, so the first attempt compared xwen's raw continuation
against llama.cpp's chain-of-thought and reported a 11.5-point chat gap that read exactly
like a graph bug. The tell was in the captured output, not the number — llama.cpp's log
carried an interactive banner and a `[Start thinking]` block. Driving the comparison
through `llama-server`'s `/completion` endpoint, which takes the prompt verbatim, and
reading `timings.draft_n` / `draft_n_accepted` fixed it. The other trap is the one the
arc already knew about and the brief insisted on: both sides must run at `p_min` 0,
because xwen's floor is a full-vocab probability and llama.cpp's is renormalized over its
top-10, so any nonzero threshold compares two different gates. (llama.cpp's `n_max` clamp
to `n_mtp_layers` applies only to multi-layer `chain_heads`, so with a 1-layer sidecar
both implementations self-feed the full depth — apples to apples, verified at
speculative.cpp:1342-1352.)

**C3 — the defaults move, and depth is what moved them.** The sweep crossed `p_min`
{0.3, 0.5, 0.7} with depth {2, 3, 4} in stage 1, 128 greedy tokens, arms interleaved,
medians of 3 reps, against a plain arm of 23.7 code / 24.0 chat. All nine arms qualified
(ahead of plain on both fixtures in every rep). The winner was **p_min 0.7, depth 4** at
33.8 tok/s mean-of-medians, against the shipped (0.5, 3)'s 32.5.

Depth is the axis that pays and the floor very nearly is not. At fixed depth 4 the three
floors spanned 33.5 / 33.2 / 33.8 — 1.8% — where depth spanned 12%. Depth 4 beat depth 3
at every floor, almost entirely on chat (+36.7 to +39.2% over plain against +27.5 to
+32.9%) while code was a wash. What the floor does clearly change is wasted work:
acceptance at depth 4 runs 65.5% at 0.3 against 80.0% at 0.7, which costs nothing
measurable at batch 1 because the target forward dominates. So 0.7 ships, and both
`hub.rs` and the record say plainly that it is the weakest-held of the three checkpoints'
floors.

Because depth 4 won at the grid's EDGE, a follow-up probe checked the optimum was
bracketed rather than merely unexplored — at `p_min` 0.7, depths 4 / 5 / 6 / 8 read
34.9 / 34.0 / 32.6 / 25.4 mean-of-medians. It falls away on both sides, so 4 is a peak.
Depth 8 is where the auto-pause controller starts firing hard (34-80 rounds paused) and
drafting stops paying at all, which is the controller working.

At the shipped configuration, measured three independent times in this one session
(stage 1, stage 2's shipped-margin arm, and the probe): **34.4-35.7 code, 33.1-34.0 chat,
against plain 23.7-24.8 — +44 to +45% and +37 to +38%**, acceptance 80.0% / 77.8%. The
cross-drafter comparison the arc wanted: the 3.6-27B's DFlash head runs 1.50x/1.47x over
its own plain arm where the 3.8's MTP head runs 1.45x/1.38x over its own, same trunk
geometry, same hour. The block drafter is still the better drafter; the MTP head closes
most of the gap for an order of magnitude less drafter KV (4 KiB/token against 40).

**A finding that was NOT installed, deliberately.** Stage 2's margin sweep made the
never-pause arm the winner — `margin 0` at 35.9 mean-of-medians against 34.8 at the
shipped 1.0. Pausing cannot explain it: both arms recorded ZERO paused rounds. The cause
is the controller's instrumentation rather than its decisions — `PauseController` forces
a plain round every 32 (and every 4 until its plain warm-up is met) to keep
`ema_plain_ms` fresh, and a forced-plain round commits one token where a drafting round
commits about four; in a ~40-round run that is about three rounds of speedup given up,
which is the size of the gap. It stays uninstalled because `pause_margin` is ONE shared
value at three sites, only this checkpoint's stage 2 was run, and decisions.md records
the controller earning its keep on the 3.6 pair — installing 0 here would silently change
two checkpoints to a value nothing graded for them, and would remove the safety net the
depth-8 arm proves still works. Ledgered as an optimization, not a default change.

**Two stage-B claims did not survive being re-run, and the ledger says so.** First, that
entry's headline "+60%, 39.3 against 24.5" was one run at the OLD defaults; the sweep's
answer for the shipped configuration is +44-45% / +37-38%, and single runs at 128 tokens
on this machine are simply not that precise. Second, and more substantively, stage B
recorded `--draft` byte-identical to `--no-draft` under BOTH equivalence modes including
sampled at 192 tokens, seed 42. It does not reproduce: the 3.8 diverges in sampled mode
at seed 42 (line 7) and at seed 7 (line 1, both fixtures), while seed 99 code and seed 1
chat come back identical.

This is not an MTP regression, and the control is what establishes that: **the shipped
DFlash 27B diverges in sampled mode too**, on the chat fixture at every one of seeds
42/7/99. Sampled divergence is the pre-existing near-tie class the script's own header
documents — the batched verify forward reassociates its f32 sums differently from the
single-token forward, and at temperature a near tie resolves to a different token. A
structural sampler-stream bug is separately ruled out: it would fire on every seed, and
two 3.8 seed/fixture pairs came back byte-identical over 128 sampled tokens, which is
impossible if the spec loop drew a different number of times than plain. GREEDY, which is
the mode that actually gates, is clean on both fixtures at 128 and at 256 tokens, and
again after the defaults moved (at the script's 0.3 coverage floor and at the shipped
0.7). What is left open is the SCRIPT's criterion, ledgered: its "a first-line fork means
the sampler stream" rule mis-grades, and its exit code presents sampled mode as a gate
that no checkpoint has ever passed.

**Harness.** `scripts/hf.ts`'s `drafter: null` for 3.8 was the single thing excluding it
from both harnesses; setting it wires up `draftingSizes()` for the sweep and the default
model list for the equivalence check at once. `retune-draft.ts` grew a `--depth-grid`
that CROSSES depth with `p_min` in stage 1 rather than sweeping it separately (fitting
each against the other's stale shipped value is how you get a half-fit), a
`SHIPPED_DRAFT_MAX` table mirroring `Model::draft_max_default`, and a status-quo
tie-break that now compares the pair. Without a depth grid the arms, labels and cell keys
are exactly what they were, so the ordinary p_min retune still measures what it always
measured. 888 tests green (819 lib + 69 binary).

## 2026-08-15 (superseded in part by Stage C above — its "+60%" is a single run at the old defaults, and its sampled-equivalence claim does not reproduce) — MTP drafting SHIPS on Qwen3.8-27B: 39.3 tok/s drafted against 24.5 plain in the same session (+60%) at 93.5% acceptance, and `--draft` is byte-identical to `--no-draft` under both equivalence modes

The checkpoint that had no drafter now has one. `src/mtp.rs` is the head (one extra
trunk-flavour full-attention layer with its own KV), `src/drafter.rs` the seam between the
two drafter kinds, and `generate.rs` the chain round. Stage A built the head and the seam
and left two `bail!` stubs where the loops go; this entry covers both stages, since A
shipped no user-visible behaviour on its own.

**Machine state: `lowpowermode 0`, on AC power, charging.** Per CLAUDE.md this machine
emits no `powermode` key, so high-power mode is not claimed — only that low-power mode is
off, which is a different state from the Phase 0 entry below (`lowpowermode 1`, on
battery) and is why none of its absolute numbers are compared against these.

**What a round does.** Draft: chain the head `min(--draft-max, 3)` steps from the token the
target just sampled. Step 1 consumes that token and the target's post-final-norm hidden at
the position before it; every later step is fully self-feeding, consuming its own previous
token and its own post-`shared_head_norm` hidden. Greedy, and a step whose argmax
probability falls below `p_min` DISCARDS its token and ends the chain. Verify, accept and
roll back are the existing DFlash machinery unchanged — checkpoint, batched
`forward_all_logits`, `accept_drafts`, `kv_rollback`, the retention cap, the auto-pause
controller — which is the whole reason a second drafter kind was affordable at all. Then
sync the head over the kept rows.

**The sync rule is the part that had to be got exactly right, and it now lives in one
function.** The head's KV row for position `p` is built from `(token_p, hidden_{p-1})` —
shifted right by one — with position 0 taking a zero hidden, mirroring llama.cpp's initial
`pending_h`. Stage A's `sync` took an already-shifted `h` and left the shift at the call
site, of which there are three (prefill, the round's plain step, the round's verify). That
was reconciled: `sync` now takes the tokens and the hiddens at the SAME positions, exactly
as a forward hands them over, and owns the shift itself, carrying the one hidden the next
row will need. `the_sync_pairs_each_token_with_the_previous_positions_hidden` pins it
against a synthetic head whose attention and FFN contribute nothing, so each row's hidden
is directly readable off the output and an off-by-one cannot hide.

**The chain stays on the GPU.** Phase 0 measured a per-step CPU readback at +1.45-2.96 ms
of pure sync, which on a 3-step chain is most of what drafting saves. So each step's argmax
and probability are reduced on device, the next step's embedding is gathered BY DEVICE
INDEX (`XwenModel::embed_rows`), the hidden is carried forward as a tensor, and one
readback happens at the end of the chain. The per-step vocabulary projection — Phase 0's
"51.8-53.6% of the step's time" — goes through the vendored seq=1 mat-vec
(`XwenModel::lm_head_row`, extracted from `forward`'s decode bypass), not QMatMul. The
`p_min` walk then runs on the host over the read-back probabilities, which means a chain
that will be cut at step 1 has already paid for steps 2 and 3; that is the trade the
on-device accumulation buys, and at depth 3 it is the cheaper side.

**Prefill pays for the head, and the cost is small.** llama.cpp pays an 84 MB-per-4k-ubatch
device-to-host round trip here because its two contexts cannot see each other's tensors;
xwen is one process on one device, so the hiddens go from the trunk's final norm into the
head without leaving the GPU. Measured as an interleaved A/B (XWEN_BENCH=1, 5 reps per arm,
alternating, medians), MTP-attached against plain: **-4.2% at 839 prompt tokens** (median 694.8 vs 725.5 tok/s,
arms spanning 691-702 and 721-734) and **-5.8% at 3286** (424.8 vs 450.9, reps 3-5 alone
430.7 vs 456.8 for 0.943 — the same ratio). Mildly worse at the longer prompt, which is
the shape the mechanism predicts: the head builds its OWN `[1, n_head, seq, pos+seq]`
prefill mask per chunk, where the trunk builds one and hoists it across all sixteen of its
full-attention layers, so the head adds a second full-size mask rather than a sixteenth of
one and the cost grows with `pos`. That mask is reusable as-is — at sync time the head's
committed length and head count both equal the trunk's — and is ledgered.

Both figures are interleaved medians and neither is a clean-machine absolute: the 4k arm's
first two reps read 237-339 tok/s against the last three's 425-460 while the machine
settled out of a concurrent test run, which is the duty-cycle effect CLAUDE.md warns about
(a doubled `XWEN_BENCH` prefill at 3286 tokens is a long run). The RATIO held across that
swing, which is the only reason the number is quotable at all. An earlier reading of -31%
at 4k was contention plus a real bug, both since resolved.

That bug is worth recording because it only appears past 1024 tokens: the head's
`LayerCache` is allocated at `max_ctx.min(1024)` and grown on demand like the trunk's, and
nothing was calling `ensure_full_capacity`. Anything over 1024 tokens of context failed
with a `slice_set` shape mismatch. The 4k prefill measurement is what caught it — the
decode smoke and every unit test run below the boundary.

**Numbers, all within-session.** First live smoke, 200 greedy-ish tokens on a code prompt
at the shipped defaults (`p_min` 0.5, chain 3): 39.3 tok/s drafted against 24.5 tok/s
plain in the same session, **+60%**, at 93.5% acceptance over 55 rounds and 3.8 tokens per
round. With the confidence floor removed entirely (`--draft-p-min 0`, chains always 3)
acceptance falls to 82.0% and throughput to 36.4 tok/s, which is the expected direction and
is what makes 0.5 the right shipped default to start Stage C from. These are single runs,
not a qualification sweep — that is Stage C.

**Equivalence holds in both modes.** `--draft` and `--no-draft` produce byte-identical
output at temperature 0 over 256 tokens, and at temperature 0.8 seed 42 over 192 tokens
with `--draft-p-min 0 --draft-pause-margin 0` — the mode that catches the spec loop
advancing the sampler stream a different number of times than the plain loop. Both were run
by hand; `scripts/spec-equivalence.ts` still has no 3.8 arm (TODO.md).

**Two shapes had to grow a kind.** The disk-tier drafter record now stores which drafter
wrote it, and `CONTAINER_VERSION` goes 2 → 3 so a v2 record is refused rather than
misread — the checkpoint binding cannot help here, since the same target file can be served
with either drafter attached. The two records genuinely differ: DFlash is f32 across
several layers, MTP is f16 across one (it reuses the trunk's own `LayerCache`) and carries
the shift-right carry besides its KV. Same for `hub::Drafter`, whose per-token cache
arithmetic hardcoded DFlash's 8 heads at dim 128: the MTP head is 4 KiB/token against
DFlash's 40-48, an order of magnitude cheaper to give context to.

**One real limitation, ledgered rather than hidden.** The head cannot follow a rewind. Row
`pos` needs the target hidden at `pos - 1` and the head keeps exactly one such hidden, so
`truncate` to anything shorter than it holds resets it to zero rather than resume on
another position's hidden — speculation goes off for the rest of that serve conversation
until a prefill from zero. The DFlash drafter keeps its rows across the same rewind because
each of its rows is a function of that position's taps alone. Three ways out are costed in
TODO.md; the cheapest correct thing was chosen for stage B because syncing on a stale carry
would be a silently wrong draft context rather than a slow one.

886 tests green (817 lib passed plus 26 ignored, and 69 in the binary; the ignored set
is the perf benches and the fixtures that need a checkpoint on disk) (`the_confidence_walk_discards_from_the_first_shortfall`,
`the_sync_pairs_each_token_with_the_previous_positions_hidden`,
`an_mtp_image_round_trips_and_cannot_be_read_as_a_dflash_one`, and the hub's kind
assertions are new). Stage C — the llama.cpp acceptance cross-check and the qualification
sweep — is a separate brief.

## 2026-08-15 — Phase 0 for MTP drafting on Qwen3.8-27B: the 3.6 DFlash head partially transfers but does not pay (0.86-1.02x vs the native head's 1.33-1.65x in the same session), and an MTP draft step costs 7-8.5% of a target forward

Measurement only — no shipped code changed. Two experiments that feed the MTP arc's design
(the arc itself is committed regardless of how they read).

**Machine state, which everything below is conditional on: `lowpowermode 1`, ON BATTERY,
discharging.** Every perf figure in this repo's "Perf state" was taken at `lowpowermode 0`.
Plain 3.8 decode read 9.0-9.4 tok/s here against the 23.8 recorded on 2026-08-14 — a 2.5x
depression — and the plain chat arm swung 9.7 → 6.5 → 9.4 tok/s WITHIN one interleaved
sweep (1.49x), with the microbench's short ops drifting monotonically slower run to run
(1.87 → 10.15 ms for the same 1 GB matvec). No absolute number here is comparable to any
other session's, and the sweep's tok/s ratios would have been worthless on their own. What
makes the experiments readable anyway is that both are built on quantities a throttled
machine cannot move: acceptance rate (a model-vs-model property), a within-session control
arm, and a bytes-moved budget.

**EXPERIMENT 1 — the 3.6-27B DFlash head DOES attach to the 3.8 target, and partially
transfers, but does not earn its keep.** The configs being byte-identical means
`check_against_target` passes and `--model-size 3.8-27b --draft <3.6 sidecar>` just works;
nothing about the mask token id (248070, which 3.8's tokenizer reads as `<|audio_start|>`)
obstructs it, because the mask is only ever fed to the drafter. Four arms, 128 greedy
tokens, XWEN_BENCH=1, the P9a code/chat fixtures, arms interleaved across reps, 3 reps,
medians — plus a NATIVE control (the same head on its own 3.6 target) and a 3.6 plain
baseline, all in the same thermal epoch, because the shipped 78-86% figure was recorded in
a different power state and comparing against it would have been the exact error CLAUDE.md
warns about.

Acceptance was deterministic to the point of boredom (50/50/50, 68/68/66, 81/81/81 across
reps), which is what makes it the load-bearing number:

| p_min | code: 3.8 / native 3.6 | chat: 3.8 / native 3.6 |
| --- | --- | --- |
| 0.3 | 50.0% / 87.4% (retains 57%) | 47.5% / 65.9% (72%) |
| 0.5 | 67.6% / 92.1% (73%) | 75.8% / 81.1% (93%) |
| 0.7 | 79.5% / 94.0% (85%) | 80.6% / 97.0% (83%) |

With the auto-pause controller removed entirely (`--draft-pause-margin 0`, p_min 0.5), so
every round drafts and the controller's timing-dependent decisions stop mediating:
code 63.7% (3.8) vs 90.6% (native), chat 69.1% vs 73.5%.

Throughput, same session: native 3.6 plain 9.2-9.3 tok/s, native 3.6 drafted 13.2 code /
12.4 chat (**1.43x / 1.33x**, and 15.2/12.7 = 1.65x/1.37x never-paused) — while the 3.8
target with the transferred head ran 8.9 code / 8.9 chat against 9.0/9.4 plain
(**0.99x / 0.95x**; 1.02x/0.86x never-paused). The controller saw it and paused 72-89% of
rounds, which is it working as designed.

The native control is what licenses the conclusion: speculation still pays 1.3-1.65x on
this machine in this state, so the transferred head's ~1.0x is a property of the transfer,
not of the battery. **The head survives the retrain partially — acceptance is far above
chance and never collapsed — but a head that proposes 64-76% where the native one proposes
81-92% does not clear its own overhead.** It is not an interim default for 3.8. What it is
is the baseline MTP has to beat, and the bar it sets is the native pair's 1.33-1.65x.

**EXPERIMENT 2 — one MTP draft step costs 7.1-8.5% of a target decode forward, and the
lm_head is half its time and 70% of its bytes.** `mtp-Qwen3.8-27B-Q8_0.gguf` (3.16 GB, 18
tensors, `general.name` "Qwen3.8-27B", `block_count 65`, `nextn_predict_layers 1`) is
structurally a 65th trunk full-attention layer. Measured through a throwaway harness over
the real weights and the shipped loader, amortized (32 dispatches per sync, outputs held
alive, warm-up batch discarded, median of 7), ctx 1024:

| | run 1 | run 2 |
| --- | --- | --- |
| (a) whole MTP step | 7.818 ms | 9.472 ms |
| (b) lm_head mat-vec alone | 4.187 ms | 4.904 ms |
| (d) lm_head + per-step CPU readback | 7.145 ms | 6.355 ms |
| (c) target decode forward | 109.395 ms | 111.506 ms |
| **step / target forward** | **7.1%** | **8.5%** |

The timing is noisy (see the machine state) so it is cross-checked against bytes, which
throttling cannot move: the step must read 451.3 MB of MTP layer weights (Q8_0) plus the
target's 1042.9 MB Q6_K lm_head = 1494.3 MB, against ~18.25 GB for a target forward —
**8.19% by bytes**, which the two timed runs bracket. Three independent routes to the same
answer. Internal consistency check: (c)'s 109 ms/token is 9.14 tok/s, matching the sweep's
independently measured 9.0-9.4 tok/s plain arm.

Two design inputs fall out. The lm_head is **51.8-53.6% of the step's time and 69.8% of its
bytes** — the chain-drafter tax dominates, so anything that shrinks the per-step vocabulary
projection is worth more than anything that shrinks the layer. And a per-step CPU readback
(sync + device→host copy, which is what a CPU-side argmax between chain steps costs) added
+1.45 to +2.96 ms/step over the same op batched, 1.3-1.7x; against a step that is itself
~2-9 ms, that is a large fraction to pay 2-3 times per draft chain. It measures the same
way in both runs and it does not shrink when the clock recovers, since it is a
synchronization cost rather than a compute one.

Verdicts as design inputs: the step-cost ratio is comfortably inside the "<10% → viable at
depth 2-3" band, and it says the draft chain should stay ON-GPU rather than reading back
per step. Every number here should be re-measured at `lowpowermode 0` before it is quoted
as a perf claim; the RATIOS and the acceptance figures are the parts expected to survive
that re-measurement, and the byte budget is not expected to move at all.

Harness (throwaway, unstaged): `examples/mtp_step_bench.rs` in-repo (untracked) and the
sweep scripts under this session's scratchpad; raw per-run JSON alongside them.

## 2026-08-14 — Review round: a job now names a FILE, not just a checkpoint; drafting is resolved per checkpoint; a contradicting `--model-size` fails at startup

Two reviews (Claude and Codex/gpt-5.6-sol) of the entry below found one shared root
behind most of their findings: the arc had introduced a second kind of model identity
(the served file's own id) without giving the engine a way to represent it. Everything
here follows from fixing that.

**A job now names a `Target` — a checkpoint plus "is this the served file" — instead of
a bare `Model`.** On a server started with a GGUF that identifies as none of the official
checkpoints, the official checkpoint of the same architecture is a DIFFERENT FILE with
the same name for sizing, and `Model` alone could not say which was meant. The
consequences the reviewers found, all now fixed: `checkpoint_paths` short-circuited to
the served file whenever `size == default_size`, so an official name was answered by
unchecked weights; the id `/v1/models` advertised for a custom GGUF was refused by the
resolver that was supposed to accept it (400 on the one id it published); and the batch
handler labeled such a run with the arch-fallback's full name, so the response document
claimed official weights had run. Equality on `Target` is file identity, which is also
what the engine's swap check and the disk tier's binding actually wanted — the tier is
bound to `settings.model`, not to a checkpoint id, and now compares as such.

**`--model-size` is a tie-break, not an override.** It settled a file that identifies as
nothing; against a file that identifies itself it silently won and then made
`EngineState::load`'s own checks fail on every request — a server that starts clean and
500s forever. It is now a startup error naming both sides, and must also agree with the
architecture. The load-time checks stay as a backstop for a file that changed under a
running server.

**Drafting is per checkpoint (promoted from a ledger item).** `ServeSettings.draft` was
one resolved `Option<PathBuf>`, so a server whose DEFAULT checkpoint ships no sidecar ran
every other checkpoint plain as well — invisibly, and worth -46 to -52% on the 27B. It is
now a `DraftMode` (`Off` / `Official` / `Custom`), resolved when each checkpoint loads.
The TUI's drafting cell follows the loaded checkpoint rather than the setting, for the
same reason. `xwen serve --draft official` on a sidecar-less checkpoint now errors like
the one-shot commands do instead of degrading quietly.

**`identify`'s file-name branch was dead code, and its live half was too loose.** It
matched `Path::file_name` (extension included) against full names, so nothing ever
matched exactly and only a bare `3.6`/`3.8` substring test fired — which would have
identified `My-Qwen3.6-14B-finetune.gguf` as the official 27B. File names are now matched
on the full name as a case-insensitive substring of the STEM, the loose release-substring
pass is kept only for `general.name` (which the converter wrote about the model it
holds), and a name matching more than one checkpoint identifies as none rather than as
whichever the table lists first.

**Also fixed:** `/v1/messages/count_tokens` ignored its `model` field, which made the
"every surface refuses an unknown model" claim false — it validates now (the count itself
is checkpoint-independent, since every checkpoint shares a tokenizer);
`retune-draft.ts --dry-run` printed `--draft-p-min undefined` for a sidecar-less
checkpoint because the drafter guard sat after the dry-run early return, and
`SHIPPED_P_MIN` is now `Partial` with a checked accessor rather than a `Record` missing a
key; two tests named `..._echoed_verbatim` asserted strings that can no longer reach
`prepare`, and now pin the real contract; the fallback drafting floor's comments stop
calling the 35B's fitted 0.3 "the shared base" and say what it is.

**New tests.** The dialect handlers are now driven directly (`#[tokio::test]` over
`probe_state`): an SDK id and both CLI aliases 400 in each dialect's own envelope with
every valid name in the message, nothing reaches the queue, a case-insensitive full name
does reach it as the right `Target`, and `count_tokens` behaves the same. `identify` gains
the blessed stems, an ambiguous name, and `My-Qwen3.6-14B-finetune.gguf`. `/v1/models`'s
test now validates EVERY listed id through the resolver, first entry included — the entry
it previously skipped is exactly where the custom-GGUF bug lived. One trap worth
recording: a non-streaming handler test hangs forever against a probe queue (the handler
waits for engine events that never come), so the servable-model case asks for a stream,
which returns as soon as the job is queued.

**Second pass over the same round.** A verification pass confirmed the ten findings above
closed and found four more, two of them introduced by the fixes themselves — worth
recording, because both were the same mistake in different clothes: a rule stated in a
comment and not checked anywhere.

**The dashboard's drafting cell was pinned OFF on every drafting server.** The cell was
made to follow the loaded checkpoint by clearing on `ModelLoaded` and setting on
`DrafterLoaded` — but `DrafterLoaded` is emitted from `attach_drafter` INSIDE
`EngineState::load` while `ModelLoaded` is emitted after that load returns, so the
clearing arm always ran last. Confirmed in a live log: `drafter loaded in 0.5s` then
`model loaded in 4.0s`. The fix does not reorder the engine (that timing is
`ModelLoaded`'s meaning): the cell is now decided only by the two drafter events and RESET
by the events that end a residency (`CheckpointSwappingOut`, `IdleUnloaded`), falling back
to a new `draft_configured` when nothing is loaded. Ordering-independent by construction.
The old test fed the events in an order the engine never produces and asserted nothing
about the cell; the new one feeds the real order through a full swap cycle and asserts the
rendered header at each step.

**The served checkpoint's official sidecar lost its startup preflight** when `validate_model`
was narrowed to custom drafters — reintroducing exactly the 500-per-request class this
round set out to kill, because "official" does not mean "fits": a custom GGUF served as
its architecture's checkpoint gets that checkpoint's sidecar. Startup now judges the
served target's official sidecar too (offline, from what the CLI's prefetch left in the
cache); other checkpoints keep attach-time checking, since their sidecars may not be
downloaded yet. Verified by manufacturing the case — an APFS clone of the 27B with
`general.name` blanked to `mymodel-27x` and `embedding_length` patched 5120 → 4096 — which
now refuses at startup with `the drafter has a hidden size of 5120 but the target has
4096` instead of starting and failing every request. The old test claimed to pin this and
only exercised `check_against_target` as a pure function; it now drives `validate_model`
itself, alongside a test of which drafter startup selects in each mode.

**`general.name`'s loose matching went too.** It kept a bare release-substring pass
("3.6"/"3.8") that the file-name pass had already been tightened away from — and since
`Arch::Moe` has exactly one candidate, no ambiguity check could save `MyMoE-3.6` from
becoming `Qwen3.6-35B-A3B` and answering official-name requests with unchecked weights.
Both sources now use one rule: exact full name, or a whole full name found inside the
name. The blessed files are unaffected (their `general.name` IS the exact full name,
verified on all three).

**Docs corrected where they had gone stale:** CLAUDE.md's cheat-sheet line and a
decisions.md paragraph still described the old "explicit `--model-size` first" precedence;
the flag is a cross-check that must agree with a file that identifies itself. The
decisions.md paragraph carries a SUPERSEDED note per that file's convention. The batch
label round-trip comment no longer claims a resubmit always works — it does not for a
custom server's own id, which names no checkpoint.

**Verification.** `cargo build --release`, `cargo fmt --check`, `cargo test --release`:
873 passed, 0 failed. Live: the full swap cycle above (35B drafting → 3.8 plain → 35B
drafting) with the event order captured; the manufactured preflight failure; and the serve
checks in the entry below, re-run after these fixes. One limitation stated plainly: the
TUI cell itself was verified through its rendering test (a real frame, asserting
`draft ON`/`draft off`) plus the live event order, not by reading a live dashboard — a
headless pty gives ratatui a zero-size terminal, so no frame text can be captured here.

## 2026-08-14 — Qwen3.8-27B added as a registry entry (same graph, no drafter), and the APIs go to full model names only: `/v1/models` stops listing one checkpoint three ways and an unknown `model` is a 400 instead of a silent default

**Two changes, one root.** Qwen3.8-27B released today; its `config.json` is
byte-identical to Qwen3.6-27B's, so it is the same forward pass over different weights
and needed no model math (the parity gate was deliberately not run: nothing it grades
changed). Adding it broke two claims the code made. `Arch::model()`'s doc said the GGUF
architecture identifies the checkpoint one-to-one — true until two releases shipped the
same dense `qwen35` graph. And the model-name vocabulary, which had been the CLI's short
aliases, could no longer be stretched: `27b` already meant 3.6 and a third checkpoint
made the aliases a naming scheme rather than a shorthand.

**What the old `/v1/models` actually returned**, from the server running on this machine
while the work was underway: `Qwen3.6-35B-A3B-Q4_K_M` (the file stem), `27b`, `35b` —
three ids for two checkpoints, one of them listed twice under two spellings. A client
picking a `model` string from that listing is picking between two names for the same
model. Alongside it, the compat dialects resolved an unrecognized `model` by falling
back to the default, so an SDK's own id was answered indistinguishably from a correct
request. Both are now one rule: the APIs speak `Model::full_name()` and nothing else —
`Qwen3.6-27B`, `Qwen3.6-35B-A3B`, `Qwen3.8-27B`, the strings ggml-org names the repos
with and the GGUFs carry as `general.name` — absent or empty still means the served
checkpoint, and anything else is a 400 listing the valid names. `--model-size` keeps the
aliases and now also takes full names, so an id from a listing pastes into the CLI.

**The identification chain that replaced `arch.model()`**: explicit `--model-size`
(threaded into `serve::run` as an `Option<Model>` so an explicit flag is distinguishable
from the default), then `general.name`, then the file name, then `arch.model()` as the
last resort with a logged warning naming what it assumed. Verified against the real
files rather than assumed: the cached 3.6 GGUFs carry `general.name = "Qwen3.6-27B"` and
`"Qwen3.6-35B-A3B"`, exactly the full names, which is why the first pass is an exact
match with a substring pass behind it.

**The drafter became optional, which was most of the diff.** Qwen3.8-27B ships no DFlash
sidecar (the repo's MTP sidecar is unread — ledger item), so `Checkpoint.drafter` is now
an `Option<Drafter>` holding file, size, layer count and fitted `p_min` together: a
checkpoint either drafts or it does not, and no caller can ask half the question. Every
consumer handles `None` by running plain with one line saying so — `ensure_drafter`,
`resolve_draft`, `checkpoint_paths` (a new `ServeLog::NoDrafterAvailable`), `xwen fetch`
(prints `drafter none`) — and `resolved_p_min` falls back to the shared base for the one
path that can still attach a drafter to a sidecar-less checkpoint, a custom `--draft`.

**Facts checked rather than trusted.** The tokenizer is NOT byte-identical between the
releases: 3.8 adds seven audio/TTS specials at 248070-248076 over an identical base
vocab (248044 entries) and merge table (247587), verified structurally after the blob
hashes differed. Text tokenizes identically, so the embedded 3.6 tokenizer still ships
and nothing was wired; whether a text-only checkpoint can emit those ids is a ledger
item, not something to improvise a second 12.8 MB embed over. 3.8's chat template DOES
differ (reasoning_effort preamble, `preserve_thinking` defaulting true, no inline
`<think>` parsing) and is vendored as `reference/chat_template-qwen38.jinja`; its
generation prompt is byte-identical to 3.6's, which is why the hand-written renderer is
unchanged, and chat.rs now cross-checks both vendored templates so a future divergence
fails a test instead of a reply.

**Verification.** `cargo build --release` clean; `cargo test --release` 869 passed, 0
failed (the two `unused_mut` warnings predate this work). The parity gate was not run and
did not need to be: no model math changed.

Qwen3.8-27B end to end, from an empty cache: `xwen fetch --model-size 3.8-27b` downloaded
18,973,870,432 bytes (the published size) and printed `drafter none (Qwen3.8-27B ships no
sidecar)`; a second run resolved from cache without a request. `xwen inspect` confirms
what the registry entry claims — `general.name = "Qwen3.8-27B"` (the exact full name, so
identification is an exact match rather than the substring fallback), `qwen35`, file_type
15, rope sections [11,11,10,0], ssm 48/16/128/6144, conv_kernel 4, eos 248046 with
`add_bos_token = false`, and all 16 `attn_output.weight` at Q6_K, which the loader parses
and the Metal load accepts without a murmur. `xwen generate --model-size 3.8-27b`: 18.3 GB
resident, 9.3 s cold load, the line `no drafter available for Qwen3.8-27B; decoding
without speculation`, coherent answer, thinking block closed, stopped on an EOG id well
inside the token budget. Plain decode measured **23.8 tok/s** (21-token prompt, 145
tokens, single greedy run, cold-ish, machine shared with other work — one run, not a
sweep; the 27B's plain figure is 24.8-25.3 under the 2026-08-08 protocol, which is the
number to compare against, and both are plain).

Serve, on a 35B-A3B default: `/v1/models` returns exactly
`["Qwen3.6-35B-A3B", "Qwen3.6-27B", "Qwen3.8-27B"]` — the served one first, each once,
against the three-ids-for-two-checkpoints listing the old binary was still returning on
this machine. `"model": "35b"` → 400 in the OpenAI envelope, `"27b"` → 400 in the
Anthropic envelope, `"gpt-4o"` → 400, `/xwen/v1/batch` `"35b"` → 400, every message
listing the three valid names. `"Qwen3.6-35B-A3B"` → 200; `"qwen3.6-35b-a3b"` → 200
(canonical echo); no `model` field → 200 echoing `Qwen3.6-35B-A3B`; batch with no field
labels its response document with the full name too. On a 3.8-27B default the same
listing leads with `Qwen3.8-27B`, startup logs `no drafter available for Qwen3.8-27B;
serving without speculative decoding`, and a request naming `Qwen3.6-35B-A3B` swapped
checkpoints and answered — the path where `EngineState::load`'s new architecture and
identity checks run against a non-default checkpoint.

## 2026-08-12 — The client's residual first-field escape report is reproduced and confirmed as conditioning signal: it follows position, not field identity, and the outside mass is almost entirely the answer in another spelling or a plausible alternative shape. No code change.

**Where it came from.** The escape-fix consumer graded 2e2280b: the 0.999 pin is gone
(their first-field median now 0.0015, in the expected range), but the first boolean
field of every multi-field item still reads 50-88× hotter on escape than its siblings
(their max 0.032 vs ~1e-5 later) — and since "first field" is whatever their TOML lists
first, it inflates per-category mean escape unevenly. They flagged it as a possible
position bias at the document's first choice point.

**Reproduced with the same instrument that settled the original bug** — a temporary
row dump at the exact `score_field` read (reverted after; scoring path untouched).
Three items, five boolean fields each, `include_score: "all"`, 505-token shared rubric
prefix, 35B-A3B default model, greedy. Items 1 and 2 classify a byte-identical email
and differ ONLY in field order (first two fields swapped) — the position-vs-identity
control the client's data can't run.

**It follows position, decisively.** No-think field-0 escapes: 0.0114 / 0.0349 /
0.0102; every later field 3.6e-6 to 1.7e-4. In the byte-identical-email control
(items 1 vs 2): swapping `urgent` out of slot 0 drops it 152× (0.0114 → 7.5e-5),
swapping `spam` in raises it 311× (1.1e-4 → 0.0349, just above the client's 0.032
max) — same email, same rubric, same options, only the order moved. Item 3, a
different email with item 1's order, independently repeats the shape (`urgent` 0.0102
at slot 0, `spam` 4.2e-5 at slot 1). The hot slot moved with the order, not the name.

**The outside mass is exactly what the 2026-08-11 entry said it was.** At every
field-0 read the top eight outside tokens carry 99.3-99.7% of the outside mass in the
no-think run (90-99.9% with thinking) and they are all plausible openers: ` "` dominates (52-73% of outside — a quoted string where a boolean
belongs), then ` True`/` False` (25-46% — the ANSWER in Python capitalization, which
would invalidate the JSON and honestly counts outside), then traces of ` {"`, ` [`,
` null`. Nothing diffuse, nothing unrelated. The contrast that is the whole mechanism (item 1's
numbers; the shape repeats on all three):
field 0's argmax is ` true` at 0.986 (space-led, INSIDE) with bare `true` at 0.0024;
by field 1 one compact `"k":v,` exists and bare `false` reads 0.9995 with the entire
outside class at 1.1e-4. Style unpinned at the first choice point, fully pinned by the
second — the 0.00197-vs-0.000051 repro pair from the fix, now at a second data point
with a different schema.

**The thinking path is exonerated, and thinking amplifies the signal.** The 2026-08-11
entry's standing suspect for field-0 anomalies (the `held` reasoning-tail
reconciliation, the one genuinely field-0-specific shape in the assembly loop) is
cleared two ways: the no-think run has no reasoning tail and shows the full elevation;
and with `thinking: true` (needs budget — max_tokens 1200 refuses because the
reasoning block can't close, 6000 runs) field-0 escape rises to 0.042-0.109 (three
raw per-item values, not an aggregate; 35B only) while
later fields stay ≤1.1e-4, with ` "` still 71-99% of outside. After 5-6 KB of prose
reasoning the model is LESS committed to the compact skeleton's style — the same
signal reading louder, not a defect.

**Verdict passed back to the client.** Not an xwen bug; a real property of the
measurement they can either embrace or column-split (their with/without-first-field
columns are the right call). One nuance ledgered as a candidate refinement: much of
the outside mass everywhere is format drift of the same answer, not value
disagreement — space-led ` True`/` False` are 25-46% of field-0 outside mass, and
bare `True`/`False` are 28-87% of the (much smaller) later-field outside — so escape
overstates value-level disagreement wherever it is large enough to matter, which
today means first fields.

**Client follow-up, same day.** Their data reproduces the think-vs-plain field-0
elevation on MEDIANS (4.7× on the 35B, 1.5× on the 27B) but not on means: the 27B's
no-think arm is heavy-tailed (max 0.109 — reaching without thinking what the 35B
needed thinking for; we never ran the 27B). Our per-item think/plain ratios
(9.2/3.1/4.1×) median to ~4.1×, corroborating their 35B figure. Their report records
medians as the figure to read; agreed — n=3 here says nothing about tails. Dump logs kept in the session scratchpad; working tree
verified clean after the revert. The 2e2280b parity-gate run is STILL pending — the
GPU was left free for the client's own benchmark runs (contended prefill reads 3× low;
see "Benching rules").

## 2026-08-11 — First client feedback lands: the scored-field escape stops lying about first fields, `shared_prefix` collapses 14 POSTs into one, the body cap goes to 100 MB, and max_ctx becomes a ceiling (lazy KV, 128k CLI default)

**Where it came from.** The first external consumer of `/xwen/v1/batch`'s scored path
reported two things and asked for a third. The bug report: on `include_score: "all"`
items with several fields, the FIRST field's `escape` reads 0.999-1.000 (18/18 of their
multi-field items) while all 142 non-first fields sit at 1e-7 to 4e-4, pinning every
category's mean escape at exactly 1/fieldCount and killing the signal — their
hypothesis was escape sampled one token early, at the skeleton's opening punctuation.
The feature ask: a `shared_prefix` field, because each item repeats the story text and
a 377 KB story against the ~2 MB body cap forced one batch into 14 POSTs, each
re-prefilling the same prefix. And the operational ask: raise that cap, raise the
context default, and make context memory demand-driven in serve.

**The hypothesis was wrong, and the bug was ours anyway.** The escape and the opener
scores are read off the same row, so a stale row would have bent the scores too. A
temporary row dump at the exact `score_field` read settled it: at the first field's
choice point (after `{"urgent":`) the model puts **54.9% on ` true` and 44.9% on
` false` — single tokens carrying a leading space** — and ~5e-5 on the bare `true`/
`false` the opener set actually contains. The model wants `{"urgent": true`; the
teacher-forced skeleton is compact; the opener-level escape read the model's preferred
spelling of the ANSWER as "none of the above". By the second field the compact document
has pinned the style (bare `false` at 99.8%) — which is why only first fields read ≈1,
single-field items included (verified: 0.9998 on a one-field item). This was the
already-ledgered formatting confound, now measured at its mechanism.

**The fix reclassifies the whole row instead of looking up openers.** `escape_mass` now
takes the full next-token distribution (`Generator::last_probs`, the full-row companion
to `last_logprobs_for`, normalized by the same code) and classifies every encodable id
by its raw BYTES (`LagunaTokenizer::decoded_vocab` reverses the byte-level alphabet,
one ~248k-entry walk cached per tokenizer): bytes that — past any leading JSON
whitespace at an unquoted field, verbatim at a quoted one — are a nonempty prefix of
some option count INSIDE whatever spelling BPE gave it; pure-JSON-whitespace tokens at
an unquoted field are FORMATTING and excluded from both sides; the escape is outside
over inside-plus-outside. (Bytes and JSON-only whitespace are both second-model review
catches, same day: per-id `decode` is lossy on a token holding part of a multi-byte
character, so a text-level match misread the canonical opener of any non-ASCII option
as escape; and `trim_start` would have discarded an NBSP as layout when it would make
the document invalid.) Measured on the repro:
first-field escape **0.9999 → 0.00197** — and the 0.00197 is real signal, mostly ` "`,
the model considering a quoted string where a boolean belongs — second field 0.00052 →
0.000051, scores bit-identical throughout (they never read the classification). The
client's mean-escape workaround can be reverted. (For the next field-0 anomaly report:
the one genuinely field-0-specific shape left in the assembly loop is the THINKING
path — a reasoning-enabled scored item prepends the decode loop's unforwarded
reasoning tail to field 0's prefill (batch.rs, the `held` reconciliation), where every
later field prepends its chosen option. Correct today and `ensure!`-guarded; look
there before suspecting the escape measure.)

**A finding the fix does not touch, ledgered.** The same row dump shows the two
spellings disagree on the near-tie: the space-channel says true at 0.55, the bare
channel (which the teacher-forced skeleton scores through) says true at 0.44. Scores
are conditional on the compact format by construction — documented, defensible, but on
near-ties the channel choice can flip the chosen value. New TODO entry with the
numbers.

**`shared_prefix` is a wire-size field, not a prefill feature.** The runner has always
prefilled the items' shared TOKEN prefix once (the LCP snapshot); what repeated was the
request body. `BatchRequest.shared_prefix` is prepended verbatim to every item's first
message before rendering, so the prompts — and therefore answers and scores — are
byte-identical to spelling the document per item (pinned by test and verified live:
identical scored output CLI-vs-CLI and over HTTP). An item with no messages fails as an
item; the serve estimate counts the prefix bytes once.

**The body cap was axum's implicit 2 MB; it is now an explicit 100 MB.**
`DefaultBodyLimit::max(MAX_BODY_BYTES)` on the API router. Verified: a 5 MB body
parses (400 on an unknown field — the parser saw all of it), 101 MB answers 413. Real
cost stays judged in tokens by the queue estimates and max_ctx, not in body bytes.

**max_ctx is now a ceiling, not an allocation.** Full-attention `LayerCache`s start at
8192 positions (`KV_INITIAL_CTX`) and double on demand up to max_ctx
(`ensure_full_capacity`: the whole old buffer copied bit-exact — not just the committed
rows, so even rows above a rewound `len` survive a growth and forward-restores keep
whatever validity the rewind discipline gave them — with one host log line per growth
step). `LayerSnapshot::Full` carries no data, so snapshots taken before a growth
restore after it — pinned by bit-exactness tests against a flat allocation, the
forward-restore case included — and the page-in path grows via `import_full_kv` while
the host-image pre-flight no longer bounds pos by allocated slots (max_ctx, checked at
model level, is the real bound). The CLI default rises 8192 → 131072 at no up-front memory cost (the 131072 ceiling is
8 GiB of KV on the 27B and 2.5 GiB on the 35B, paid only if a prompt gets there; load
allocates 0.5 / 0.16 GiB); serve keeps its 262144-clamped-to-trained default, now
meaning "up to". A growth pass ends in one device sync, since candle's pool frees the
replaced buffers only there — without it the top doubling step transiently holds old
and new allocations together (~1.5x). Idle unload is
the reset: dropping the model drops whatever the cache grew to, and the reload starts
at 8192 again. Verified live: an 11215-token prompt on the new default logged one
growth step (8192 → 16384, 0.3 GB) and answered correctly.

**The TUI stops lying about batch jobs (same day, operator feedback).** Watching the
client's workload live exposed two dashboard failures. The NOW pane said "deadline in
96m15s" — the watchdog kill-ceiling, computed from the queue's deliberately-loose
bytes/3 estimate (731,183 "tokens" of a ~2.2 MB body ÷ the 150 tok/s floor rate + the
summed budgets ÷ 10 + slack), rendered as if it were an ETA; it now reads
"watchdog in …" and the prompt figure carries "~…(est)" when it is the estimate
(`JobPicked.estimated`, true for a batch, whose text is untokenized at pickup). And
HISTORY rows for batches read `321,930+0→2,252  —  0.0s  —  —`: unlabeled columns,
`prefill_tokens`/`*_secs`/`stop` never filled by the batch path — so the flow read as
"322k from cache, nothing prefilled", the duration as instant, and outcome/rate as
nothing. The record now carries a `BatchSummary`, `run_batch_job` fills the token arithmetic in
the record's own `prompt = cache_read + prefill` terms (the shared prefix is prefilled
once and READ N−1 times, so the summed per-item `cached_prefix_tokens` overcount by
exactly one span), and the table grew a header row. The first cut blended everything
into one `output ÷ whole-run` rate; that was called out as misleading (correctly — it
described neither machine) and replaced the same day by MEASURED per-phase columns:
the batch runner now reports what it actually forwarded and what it actually decoded
(`BatchStats.{prefill,decode}_{tokens,ms}`, on the wire for clients too — prefill by
delta off a new cumulative `Generator::prefill_spend` counter, since the shared prefix
and the scored trials belong to no single item; decode straight off each item's
`DecodeOutcome`), and HISTORY shows `prefill` and `decode` as separate columns, each
with its own time and its own t/s, with TTFT/spec%/whole-run total in a `detail`
column. A fully scored batch decodes nothing and its decode cell is honestly a dash —
its work shows up as prefill (teacher-forced trials), which is what it is. The LOG pane's follow latch also stopped being one-way:
scrolling used to disengage follow permanently (only End re-armed it), so a pane once
touched sat pinned to a stale line forever; scrolling back down ONTO the newest line
now re-engages follow — arriving at the live edge means wanting it live.

**Also swept while in there.** Three comments still carried laguna's ~70GB checkpoint
into a repo whose files are 19-20 GB (`MEMORY_WARN_BYTES` doc, the config template's
`idle_unload` note, gguf.rs's residency measurement — the last now attributes its ~10%
figure to laguna's working set explicitly). The parity-gate script's refusal message
still says it; that sits under the standing "scripts/ maxuna-isms" ledger item.

**Result.** 791 + 69 lib/integration tests pass, 0 fail; new tests cover the escape
classification (6, byte-level and NBSP cases included), the byte table's alphabet
reversal (1), shared_prefix (2 + the estimate), and KV growth (3, forward-restore
across growth included). Three-perspective review, each catching something the others
did not: the in-family reviewer found the scheduling estimate assuming the prefill
dedup always fires (now per-item, erring loose for the watchdog); the serve explorer
found three wrong figures in fresh comments, the growth-pass pool peak (candle frees
replaced buffers only at a sync — `grow_kv_capacity` now ends in one) and the load-
bearing import-before-restore order at the page-in site (now commented); the
second-model pass (codex `gpt-5.6-sol`) found the byte-level lossiness and NBSP
classes above plus the buffered-body concurrency exposure (accepted for loopback,
ledgered). The parity gate was attempted per the standing rule (the growth path
touches the forward, if not the math) and refused to start: a foreign `xwen serve`
process holds the GPU. Pending on a free GPU; everything else in this entry is done.

## 2026-08-11 — `/xwen/v1/batch` ships and the server stops being pinned to one checkpoint: every request names its model, the engine swaps lazily

**Where it came from.** The batch runner had been CLI-only since it shipped
(2026-08-09), with the HTTP endpoint parked behind P10 in the ledger. The ask that
unparked it also reshaped it: the endpoint should not be pinned to the served model —
`--model` should only decide which checkpoint is the DEFAULT for the compat dialects,
and any checkpoint should lazy-load and idle-unload like the one the server started
around. That turns "add a route" into "make the engine checkpoint-aware", and the route
falls out of it.

**What shipped.** Two things, one mechanism:

- **The engine is checkpoint-aware.** `EngineState` now records which official
  checkpoint it holds (`size`), every queued job names the checkpoint it needs (a new
  `Job` enum over generations and batches), and the pickup compares the two: on a
  mismatch the live conversation is imaged out through the same path an idle unload
  takes, the state drops, and the lazy load brings in the named checkpoint — one model
  resident at a time, which is this machine's memory invariant. Which checkpoint the
  default GGUF is comes from its `general.architecture` via the new `Arch::model()`,
  never from `--model-size` (the flag and a `-m` path can disagree; the file cannot).
  A non-default checkpoint hub-resolves lazily (download on miss, logged), speculates
  with its own official sidecar at its own fitted `p_min` — `draft.p_min` left unset
  now stays `None` through the config merge and resolves per loaded checkpoint at
  attach time — and runs without the disk tier, which stays bound to the default
  checkpoint (verifying it against foreign weights would permanently disable it, and
  feeding it foreign images would poison the store; decisions.md "Serving").
- **`POST /xwen/v1/batch`** takes exactly the JSON document `xwen batch` reads on
  stdin and returns exactly the document it prints. Its `model` field is honored per
  request (absent = the server's default; unknown = 400, the CLI's strictness). The
  job rides the ordinary `JobQueue` as the second `Job` variant, scored and
  deadline-bounded by a bytes/3 overestimate so small chat requests keep jumping ahead
  of it, and answers with a single terminal `EngineEvent::BatchDone` carrying the whole
  response. The compat dialects gained the lighter half of the same feature: a `model`
  that parses as a known checkpoint ("27b"/"35b") selects it, anything else falls back
  to the default and is echoed as before, and `/v1/models` now lists the default id
  plus both selectable names.

**The runner grew hooks instead of opinions.** `run_batch`'s two stderr `eprintln!`s
became a `BatchProgress` callback (the CLI prints byte-identical lines; the server logs
`ServeLog::BatchProgress`), and a `cancelled` poll threads client-gone/deadline/
shutdown from the job's cancel token between items and per decoded token — with a latch
in `run_item` so an item whose decode was actually cut reports the cancellation in its
own `error` field while an item that finished just before the signal keeps its true
answer. A cancelled batch therefore still returns a truthful partial document to a
client that is owed one (deadline), and costs nothing to one that is not (gone).

**Cache discipline around a batch.** The batch runner owns the whole KV cache while it
runs (its shared-prefix snapshot is its own machinery, not the slot manager's), so
`run_batch_job` pages the live conversation out first — it survives warm in its slot's
host image, and on disk when the tier is on — and marks the engine dirty before the
first runner call; the existing post-job reset is what puts the cache back. A batch on
the same checkpoint therefore costs the next chat turn a page-in, not a re-prefill,
and a batch on the other checkpoint costs what the swap costs (~3 s load plus the
outgoing checkpoint's warm slots).

**Verification.** `cargo test --release` green: 780 + 69 passed, 0 failed (25 + 3
ignored as usual), including new tests for the endpoint's request shape (the CLI's,
`deny_unknown_fields`), per-request model resolution (named wins / absent means the
server default / unknown 400s), the queue-side size estimates, the config merge
leaving `p_min` unresolved unless pinned, and the scheduler's per-checkpoint discount.
No model math changed, so the parity gate was not re-run. The clippy warnings the arc
introduced were fixed (both `Job` variants are boxed for the queue's sake); the
pre-existing ones were left alone.

**Review (same day).** A two-family pass — a Claude reviewer with two corroborating
explorers, plus a Codex second-model review — confirmed the swap/dirty/disk-tier
architecture sound (disk gating airtight, no unlink-of-fresh-image path, no
sampler/grammar leakage, `state.size` cannot disagree with the loaded weights) and
surfaced six fixes, all landed within the arc. The one real correctness bug was NEW to
the arc, opened by exactly what the arc did: (1) **a chat job's thinking budget leaked
into the next batch job** — `run_job` arms `set_max_think` per generation but the
batch runner never touched it, and before this arc a batch always ran in a fresh CLI
process where the leak could not exist; a stale ceiling would fail every batch item it
did not fit and silently truncate the reasoning of the ones it did. `run_batch` now
clears both thinking controls up front. The rest: (2) an absent `model` on the HTTP
batch would RUN on the server's default but be LABELED with the CLI's compile-time
default — the handler now writes the resolved name back into the request before
queueing, so runner, job and response label cannot disagree; (3) a batch item cut
mid-decode kept its partial text beside `error`, contradicting `ItemResponse`'s
"failed items carry error and empty text" contract — the text is now dropped, usage
still counts the spent tokens; (4) the spec loop polled `should_stop` right after
committing its first token with no budget guard, so a `max_tokens: 1` item — the
single-token classification shape — whose deadline fired at exactly that instant
reported a complete, correct answer as cancelled; the poll is now guarded by
`decoded < max_new` like its sibling, AND (Codex caught the residual half) `run_item`'s
cut latch now defers to `hit_eog`: a decode that ended naturally — EOG, or a
single-token grammar value completing on its first draw — is a complete answer
however late the cancellation landed, on both decode paths; (5) `run_serve` resolved the official drafter
sidecar from `--model-size` before ever opening the GGUF, the last holdout against
this arc's file-is-authoritative rule — it now reads the served file's architecture
first (the one-shot CLI commands keep the flag's double duty deliberately); (6) the
scheduler's cache discount was blind to checkpoints — both models share a tokenizer,
so a long conversation re-sent for the non-resident checkpoint could ride a
token-level match against the wrong model's warm slots to the front of the queue,
then pay a swap plus a cold prefill; the cost closure now sees the whole job and gives
no discount across checkpoints. The per-checkpoint `p_min` resolution was extracted
into a testable helper (`resolved_p_min`) with a unit test, per the review's coverage
note. The Codex pass (gpt-5.6-sol) independently re-found the think-budget leak and
the cross-checkpoint discount, verified five of the six fixes complete, caught the
grammar-completion residual folded into (4) above, and added three smaller closes:
the compat dialects now parse the CLIENT's own model string rather than the
default-substituted echo (a GGUF someone names `35b.gguf` must not route model-less
requests by its basename), the batch size estimates saturate instead of overflowing
on hostile budgets, and the checkpoint-swap log line no longer claims imaging that
only happens when the disk tier serves the outgoing checkpoint. Nits documented
rather than coded around: the `--init` template's commented `p_min` line warns it is
the 35B's value; hf-hub's raw-stderr download bar under `--tui`, the route's inherited
~2 MB body limit, batch prefills not polling cancellation, and the zero-cost estimate
for schema-only items all joined the TODO ledger. Final count after fixes: 781 + 69
passed, 0 failed.

**First live exercise (same day).** `scripts/classify-demo.ts` was switched from
spawning `xwen batch` to POSTing `/xwen/v1/batch` — against a server already running
on the default port when one answers `/health` (a hazard guard as much as a
convenience: spawning a second server beside a running one would resident two ~20 GB
models), `--url` for anywhere else, else a server it spawns on a free port and tears
down. Its first run went against a live `xwen serve` and exercised the whole arc in
one process: the 35B lazy-loaded cold (6.1 s) for the first batch, the second batch
named the 27B and the engine swapped (14.5 s — 27B weights plus its 3.5 GB drafter
sidecar, cold), and both reports came back with the demo's documented accuracy shape
(35B 8/12 with the known near-tie misses, 27B 10/12), shared prefix 317 tokens cached
across all items on both. `--no-draft` now applies only to a server the script spawns;
against a running server, that server's config decides and the script says so.

## 2026-08-09 — `xwen batch`: one prefill for N items, and scored fields that report the model's confidence instead of a sampled token

**Where it came from.** Two research passes opened the arc — one on the
classification/structured-extraction state of the art, one on serving-side prefix reuse
(BatchLLM, Marconi, SGLang's RadixAttention). They converged on the same shape from
opposite directions. The serving literature says the cheapest batch is one that groups
by shared prefix and prefills it once; the classification literature says the accurate
way to label a document along nine taxonomies is nine narrow questions, not one wide
one, because a single prompt asking for nine fields lets early fields condition later
ones and gives no per-field confidence at all. Decomposed classification is exactly the
workload whose items agree on everything but their last sentence. So the batch API and
the classification surface are one feature, not two that happen to share a subcommand.

**What shipped.** `xwen batch` reads one JSON request on stdin and writes one JSON
response on stdout; the checkpoint comes from the payload (`"model": "27b"` / `"35b"`)
because one request is one model's work. It renders every item, computes the longest
common prefix of the rendered TOKEN vectors, prefills it once, takes a
`take_cache_snapshot` in RAM (62.8 MiB on the 35B, 149.6 MiB on the 27B, and
position-independent), then replays it per item: restore, `sync_drafter_to`, prefill the
item's own tail, decode. The cache is request-scoped — no disk, no TTL, no eviction, no
tree. Per item: `messages`, an optional JSON schema, `thinking` (`false` / `true` / a
string to inject), an assistant `prefill`, `max_tokens`, `sampling`. A failing item
lands as an `error` on its own response and the batch carries on, because the entire
point is that one bad item must not cost the other N−1 their prefill.

**Three floors, and one is a correctness precondition wearing a heuristic's clothes.**
A prefix under `MIN_SHARED_PREFIX` (64) is not worth the bookkeeping and every item runs
cold. A single-item batch never snapshots. And the shared span is capped at
`min_len − 1`: if the LCP were allowed to equal the shortest item's whole length, that
item would decode from the snapshot position and write into positions the snapshot
covers — which `restore_cache_snapshot` forbids by construction (model.rs:716; the
full-attention layers restore by truncation and are only valid while the slots below
the position still hold the same keys). The cap is that precondition written as
arithmetic, not a safety margin.

**One design decision reversed its own premise.** The plan called for snapshotting at a
chunk-64 boundary, inherited from Marconi — where it is right, because a chunked prefill
kernel cannot materialize recurrent state mid-chunk. xwen has no such constraint: the
generation layer materializes full state wherever a prefill stops, which is what serve's
`PrefixCache` already depends on. Rounding down would have meant writing alignment logic
in order to throw away reuse that costs nothing to keep. The argument that recommended
the boundary — keep it simple — is the argument that killed it.

**The scored path: the engine writes the JSON, the model only ranks the values.**
Annotate an enum or boolean property with `include_score` and the item stops
free-decoding entirely. The runner teacher-forces the JSON skeleton it wrote itself and
scores every allowed option at each field's choice point by teacher-forced sequence
logprob; selection is argmax at temperature 0, posterior sampling honoring `top_k` and
`top_p` otherwise. The field reports `{"value", "score"}`, or with `"all"` the whole
`scores` table plus the `escape` mass that opened no option. The alternative — keep the
grammar and read confidence off the mask at the grammar's choice points — died on
llguidance 1.7.6 having no structural-position API: no captures from JSON-schema
grammars, and capture-on-completion semantics anyway. Assembly makes the selection exact
instead of inferred and needs no llguidance change. `include_score` PRESENCE routes the
item, so `false` is an annotation like any other and does not mean absent — a schema
cannot switch engines by toggling a boolean. New public API for it:
`Generator::last_logprobs_for`, a log-softmax over an encodable slice, in f64. Free
thinking still works on a scored item: it decodes until `</think>` and then assembles.

**The turn of the story is a finding from the outside-model review.** codex
(`gpt-5.6-sol`) pointed out that scoring an option's own tokens and nothing else makes a
strict-prefix option unbeatable: score is a sum of log-probabilities, every additional
token contributes something negative, so `score("low_priority") = score("low") +
(negative)` and `"low"` wins whatever the model believes. The fix is to score one
terminator token past the value — the closing quote for a string enum, the delimiter
after a bare literal — which forces the two sequences to diverge at a token they cannot
share and makes the option set prefix-free. Confirmed live on the adversarial pair
`["low", "low_priority"]`: the long option takes 0.9988 when the prompt asks for it. The
same pass hardened the seams: plan time now re-encodes every (segment, option) and
(option, segment) pair and refuses a schema whose tokenization is not the concatenation,
which rejects values that fuse with their delimiter and values needing JSON escapes.
Refusing beats scoring an escape sequence as if it were the label.

**The demo, and what it measures.** `scripts/classify-demo.ts` classifies one support
email along nine taxonomies — eight single-field items plus one compound, ten labels in
all — against embedded ground truth, on both checkpoints in sequence (one large model
process at a time). 317-token shared prefix, thinking off, greedy, `lowpowermode 0`
(this machine exposes no `powermode` key, so high-power mode is never positively
confirmable and is not claimed here):

```
35B-A3B   7/10 vs authored ground truth   total 1828 ms   (load 3470 ms)
27B       9/10                            total 6595 ms   (load 9859 ms)
```

The cache earns its keep on the same request: 1203 ms against 1989 ms with
`XWEN_BATCH_NO_CACHE=1` (an earlier run of the same payload; identical on every decisive
field). The misses are the more interesting half of the reading, and they are the reason
scores exist — the 35B's sentiment miss carries 0.56 on `mixed` against 0.43 on
`negative`, and its emotion field is a 0.51/0.48 coin flip. A consumer that thresholds
on score routes both to review instead of trusting them.

**Two numbers that are not stable, stated as such.** Replay and scratch decode the same
values but not the same bits: an item's short tail prefill routes the MoE through
`mv_id` where one long cold prefill takes `mm_id`, and across the demo batch the scores
differ in the third to fourth decimal, with one genuine near-tie (0.502 vs 0.493)
selecting differently between arms. That is the partition-dependence class decisions.md
already records, reached through a new door.
`XWEN_MM_ID_MIN_SEQ=1` forces both arms onto one kernel and makes them byte-identical,
which is how it was settled. And `escape` is honest only where the choice point is
quoted: for a bare literal the opener competes with pretty-printing whitespace, and a
boolean field was observed reporting 0.998 escape beside a 0.9986 answer. Kept,
documented on the field, refinement ledgered.

Decisions in decisions.md "Batch" (nine entries: the exact-LCP snapshot, single-level
request-scoped caching, the three floors, drafter-by-truncation, the greedy/no-thinking
defaults, assembly over grammar-mask detection, terminator-token scoring, the seam check,
the escape confound, and the score-stability class). Ledgered under "Deferred from the
batch + scored-classification arc": the serve `/xwen/v1/batch` endpoint, multi-level
prefix trees, streaming, per-token logprobs, a formal Track-B case for
snapshot-replay-vs-scratch, and the v1 schema limits.

**UPDATE, later the same day: the demo grew a multi-select taxonomy, and it produced the
arc's own thesis as a measurement.** Tagging (multi-select from a ten-tag set) went in
twice behind the same shared prefix — once as an array-of-enum through the grammar path,
once as the identical tag set expressed as a flat all-boolean scored schema, one
`include_score` P(true) per tag. Two forms of one question, so the comparison is
controlled by construction. Ground truth changed in one place first: `emotion_intensity`
was amended medium → high, because both checkpoints disagreed with the authored label
and the author accepted the models' reading — recorded because the grade moved with it,
and a graded row that quietly changed definition is worse than a miss.

Twelve graded rows now, same machine and same caveat as above (`lowpowermode 0`, high
power not claimable here):

```
35B-A3B    8/12   total 3416 ms
27B       10/12   total 10591 ms
```

**The result worth the addendum: on BOTH checkpoints the scored boolean vector beat the
free-decoded array on the identical tag set, 5/6 tags against 4/6.** Per-tag scoring
recovered `refund_request`, which joint array emission dropped on both models — the
failure the decomposed-classification research predicted (a model emitting a list
conditions each element on the ones before it and stops early), reproduced inside the
demo the arc shipped to argue for decomposition. The one remaining miss is NOT a
decomposition failure and should not be read as one: `replacement` scores P(true) 0.15
on the 35B and 0.26 on the 27B, and the tag rules never disambiguated the word — ground
truth means "a replacement occurred", both models read "a replacement was requested",
and the email's author is explicitly asking for a refund INSTEAD of another replacement.
Low confidence on a genuinely ambiguous label is the scored path behaving correctly.
Amending the tag rules would move the number; it is left alone deliberately, because the
row is the clearest live example of a score that should be low.

**And then `--compare-thinking`, which turns the demo into a controlled experiment on
reasoning itself.** One 22-item batch per model: every taxonomy twice, once with thinking
off and once with a force-injected closed `<think>` scaffold. The scaffold restates the
rubric and the evidence discipline and never states a document fact — it is a reasoning
protocol, not a hint, which is the only way the comparison stays about thinking rather
than about a leaked answer. Both arms sit behind the same shared prefix, so 22 items are
still one prefill and one snapshot. 6.4 s on the 35B, 21.1 s on the 27B, same power-mode
caveat as everything above. Note what the scaffold costs in output: nothing. Injected
reasoning is prompt-side, so the thinking arm spends zero completion tokens on it.

**The scaffold moved exactly the rows the research said it would.** Both of the 35B's
near-tie misses flipped to correct — sentiment `mixed` → `negative` at 0.56 → 0.76,
emotion `frustration` → `disappointment` at 0.51 → 0.69 — and every row that had already
scored 1.00 was left untouched. That is the thinking-helps-ambiguous-rubrics-only pattern
reproduced in-demo, and the scores are what make it legible: the confidence rose where
the answer changed and did not move where it did not.

**It also flipped `emotion_intensity` high → medium on BOTH checkpoints, confidently
(0.86 and 0.89).** That label has now gone three ways across three protocols — authored
medium, amended to high on the models' agreement, and back to medium under the scaffold.
Three readings from one document is a rubric problem, not a model problem, and it is the
right conclusion to draw before the tempting one: an underspecified label does not become
specified by being graded more times. Net accuracy therefore splits: 35B 8 → 9, 27B
10 → 9.

**And then it resolved: the author flipped expected `emotion_intensity` back to medium,
because the scaffolded reading won the argument on its merits.** So the label went out
the way it came in, having been argued into and back out of `high` — the rubric problem
above is real, and this is what settling it looks like rather than a fourth reading. The
predictions did not move; only the key did, and the re-grade was verified live. Under the
corrected key injected thinking wins on BOTH checkpoints: **35B plain 7/12 → thinking
10/12, 27B plain 9/12 → thinking 10/12.** That supersedes the 8 → 9 / 10 → 9 split in the
paragraph above, which was correct against the `high` key and is kept because it is what
the split looked like at the time — and it strengthens the conclusion it was hedging: the
scaffold does not trade one model's accuracy for another's, it lifts both.

**Closing the `replacement` thread: the rule was sharpened after all, and the ambiguity
resolved exactly where its probability said it would.** The paragraph above left the tag
rules alone on the grounds that a low score on an ambiguous label is the scored path
working; that held right up until the rule was made to say what it had always meant —
"sent, offered, or requested at any point, even if the customer now declines further
replacements". `tags_scored` is now 6/6 on both checkpoints and in both arms, with
`replacement` joining the confident mass rather than merely creeping over a threshold.
The free-decoded array over the identical tag set STILL drops `refund_request`, on both
checkpoints and in both arms. That is the finding to keep: the decomposed-over-joint gap
does not come from a fuzzy rubric and does not close when the rubric is sharpened — it is
structural, a property of emitting a list versus scoring each member. Final numbers (same
power caveat; 6.4 s on the 35B, 21.5 s on the 27B), superseding the re-grade above:

```
35B-A3B    plain  8/12    thinking 11/12
27B        plain 10/12    thinking 11/12
```

The thinking arms remain identical across the two checkpoints, and the only miss anywhere
in either thinking arm is the joint-array control item — the one the experiment exists to
lose. `scripts/classify-demo.ts` carries these as header comment blocks, written as
properties rather than as results, so the file states what it demonstrates without dating
itself to this run.

**The most striking result is not accuracy at all — under the scaffold the two
checkpoints agree on every single field.** The cross-model diff is empty. Injected
thinking is acting as a variance reducer here, and it lifts the fast 35B to the slow
27B's level of agreement, which is a far more interesting property for a batch
classification surface than either model's raw score. Per-taxonomy scaffold selection is
the protocol this indicates — scaffold the ambiguous rubrics, leave the saturated ones
alone — but that is a hypothesis fitted on one document, and adopting it needs held-out
validation rather than selection on the same email it was read off.

## 2026-08-09 — `--stats` splits decode by round class: plain / drafted / full-accept, with an in-run plain baseline and an estimated net drafting effect

**The old block was arithmetically correct and read backwards.** `draft` and `verify`
both divided by ALL rounds, so a run whose controller paused 93% of its rounds printed
`verify 12ms/round` next to a draft cost averaged over rounds the drafter never ran.
That reads as "verification is slower than plain decode", when per committed token the
verified rounds were the cheaper ones and the drafter had barely fired. Nothing in the
block answered the question the block exists for: did drafting help THIS run, and by how
much.

**What ships.** `SpecStats::bucket_round` folds every round into one of two classes as
the loop finishes it. **Plain**: no draft block was verified — paused rounds,
empty-draft fallbacks, serial thinking rounds, rounds past the drafter's context — one
committed token each. **Drafted**: a block was verified; the round's full wall time,
draft phase included; tokens are accepted plus the bonus. **Full-accept** is a subset of
drafted, the rounds where every drafted token survived, which is the ceiling a longer
block could reach. Two counters come with it: `spec_draft_ms`, so `draft_ms -
spec_draft_ms` is drafter time spent on rounds that ended plain, and `draft_rounds`, the
rounds the drafter forward actually ran. One shared bucketer is called from both the CLI
loop (`generate_spec`) and serve's (`decode_loop_spec`) — serve's printer is unchanged
and feeds the bucketer its own emit-excluded `round_ms`.

**The design decision worth naming: the plain bucket excludes wasted drafter time.** It
folds `round_ms - draft_ms`, which is exactly the quantity the pause controller's plain
comparator folds. It is deliberately not "what the round cost" — it is what a plain
decode step costs on this run's text, so `plain_rounds / plain_ms` is an interleaved
plain baseline measured inside the run being reported. That immunity to between-session
drift is the point: the entry below has to warn against differencing a drafted 27B
figure against a plain one from another session, because the 27B's level moves. This
baseline never crosses a session boundary.

**Derived lines.** `drafting: X.XXx vs plain on drafted rounds` is the drafted bucket's
token rate over the plain rate. `est. net ±Y.Y%` prices every committed token at the
run's own plain rate and compares that against the decode time actually spent — plain
forwards, plus full drafted rounds, plus the wasted drafter ms — three terms that
partition decode-loop model time with no double counting. Both need `plain_rounds >= 8`;
a handful of plain rounds is noise, not a rate. Zero-round buckets print nothing (a
margin-0 run has no plain rounds, a fully paused one no drafted ones). The misleading
denominator is fixed in the same pass: `draft` now averages over `draft_rounds` and says
`ms/draft`. `bucket_round_partitions_time_and_tokens` pins the partition offline.

**What it looks like** (35B, thinking-heavy code prompt, smoke run, `lowpowermode 0`):

```
decode:  768 tokens in 7.82s (98.2 tok/s)
         plain:         703 tok in 7.09s (99.2 tok/s over 703 rounds)
         drafted:        64 tok in 0.72s (88.5 tok/s over 16 rounds, 4.0 tok/round)
         full-accept:    31 tok in 0.25s (123.2 tok/s over 6 rounds)
         drafting:    0.89x vs plain on drafted rounds; est. net -1.0% overall
spec:    719 rounds (701 paused), 95 drafted, 48 accepted (50.5%), 47 rejected
         814 verified positions; draft 0.2s (10ms/draft), verify 7.7s (11ms/round)
```

One run, not a benchmark — but it is the shape the instrument exists to make visible:
701 of 719 rounds paused, and across the 16 that drafted the token rate came in below
this run's own plain rate.

## 2026-08-08 — the small-batch window reaches the attention and DeltaNet projections (verify span 8 −12.0%), and the first real controller sweep makes `draft_p_min` per-checkpoint: 0.5 on the 27B, +11-13% over the shipped 0.3

**Two arcs, and the second is the one the first predicted.** The entry below shipped
`mul_mv_ext` behind `QLinear::forward` and closed on two things: a list of sites the
window does NOT cover, and the observation that the controller had already started
behaving differently because verifies got cheap. This entry covers both follow-ups —
extending the window to the q8_0 attention and DeltaNet projections, and running the
`p_min`/`pause_margin` retune that unblocked.

### Arc 1 — `Proj::DenseF16Q8` joins the window

**What shipped.** The q8_0-stored attention projections now take the already-vendored
q8_0 `mul_mv_ext` kernel at seq 2..=8, where they previously ran the single-token gemv
once per token. One `Proj` variant covers seven tensors across every layer of both
checkpoints: `attn_q` / `attn_k` / `attn_v` / `attn_output` on the full-attention layers
(16 of the 27B's 64), and `attn_qkv` / `attn_gate` / `ssm_out` on the DeltaNet layers
(the other 48, through the same type from `linear_attn.rs`). `AttnQ8` grew a
`QuantPlane` VIEW over the buffer and `base_off` the gemv already used, so the second
kernel costs no extra memory — it reads the identical mmap alias.

**The gate is the `QLinear` gate, threaded verbatim**, which is the point: `mv_ext_window`
supplies the plan, `mv_ext_supported` checks dtype and reduction width, and
`XWEN_MV_EXT_CLASSIC` reverts this site exactly as it reverts the others. Two things
differ here and both are documented at the call site. `XWEN_MV_EXT_MAX_SEQ` cannot widen
this site past 8 — the enclosing `Q8_DECODE_MAX_SEQ` arm has already sent seq > 8 to the
dense f16 plane — so the probe knob only moves the `QLinear` sites. And there is a
16-byte activation-alignment guard, because the ext kernel reads the activation as
`float4` while the gemv takes any offset. Every production activation reaching this
window today is offset-0, so the guard has never been observed to fire; the comment says
what to do before that changes, which is to record what ran, since the `mv_ext`
provenance field is env-derived and structurally cannot see a per-call fallback.

**Deliberate skips.** `ssm_beta` / `ssm_alpha` are dense f32 at `[5120,96]` — nothing to
stream. The MoE routed experts belong to the `mv_id` family, and `XWEN_MM_ID_MIN_SEQ=1`
is already refuted at these spans. The `seq == 1` lm_head bypass stays on its current
path because it is a strict-tier bitwise anchor. And the f16 ext variant ggml
instantiates was not ported: the q8_0 alias over the same weight streams half the bytes,
so porting f16 would add a kernel that is strictly worse at every site that has both.

**The accuracy result is level here, and that is a different claim from the entry below.**
At the `QLinear` sites `mv_ext` replaced candle's mm and came out 20-400x closer to
exact, because the mm stages weight tiles as half. At these sites it replaces the
vendored q8_0 gemv, which narrows nothing either — both multiply raw int8 by an f32
delta and accumulate in f32 — so the two sit in the same ~1e-6 class and their ratio is
reduction-order noise, measured within 1-2% of each other with the direction varying by
shape. The new test constant is `GEMV_MULTIPLE = 2.0` rather than the `CLASSIC_MULTIPLE`
1.0 used against `QMatMul`, wide enough not to grade noise and narrow enough that an f16
staging path creeping back in (~1e-4, two orders out) still fails it. The `mv_ext`
provenance doc-comment in `tests/parity.rs` was corrected in the same pass: "never the
further from exact" was only ever true at the `QLinear` sites.

**Two-model review.** The Claude reviewer found nothing and independently confirmed the
alignment-guard arithmetic and that no production activation is strided. Codex
(gpt-5.6-sol) raised no Critical or High findings; its two Low findings and one Nit are
all fixed — the `MAX_SEQ` exception note, a `base_off != 0` assertion plus an
`XWEN_LOAD_CLASSIC` skip in the routing test, and the parity.rs doc-comment above.

**Both parity gates ALL PASS, at pre-change numbers** — 27B strict cos 1.000000, mm
1.000000, decode 64/64 on all three fixtures, Δnll 0.000243; 35B mm 0.999631, Δnll
0.000791. No schema change was needed: `mv_ext` records an env-derived mode, not a site
list, so v8 still describes this dump correctly.

**Measured** (interleaved A/B, `spec-verify-bench` on the 27B at `n_past` 512, this
binary against a HEAD-commit binary built in a scratch clone; `lowpowermode 0`; 5 reps
per arm pooled from two A/B sessions, medians. The first session had a `cargo` compile
overlapping part of it, the confirmation session was clean, and the two agree — the
pooled numbers are what to quote):

| span | HEAD | with Proj coverage | |
|---|---|---|---|
| 2 | 61.47 ms | 62.88 | +1.4 (+2.3%) — see the ordering caveat |
| 4 | 87.66 | **83.32** | −4.3 (−5.0%) |
| 6 | 140.39 | **131.92** | −8.5 (−6.0%) |
| 8 | 175.20 | **154.19** | −21.0 (−12.0%) |
| 12-48 | — | — | ext inactive; arms within 2.4% |

Span 8 is the cell to trust most: the two arms' per-rep ranges do not overlap at all
(165.9-183.4 against 145.2-159.3).

**The ordering caveat, which changes what the span-2 cell means.** The interleave ran
`old, new, old, new, …`, so the coverage arm is second in every pair. At spans 12, 16 and
24 — where the kernel provably cannot run — the second arm still reads slower in all five
pairs, with pairwise medians of +2.3% / +2.0% / +1.6%. The span-2 pairwise median is
+2.8%, the same magnitude, and its pair-to-pair spread is far wider (−11.9% to +5.6%).
**So span 2 is a wash on this data, not the small regression it looks like in the
table**, and the wins at 4-8 are if anything understated by roughly two points. The mechanism that would explain a genuine span-2 loss is real
enough to keep as an option — at t=2 the ext kernel saves only one gemv weight pass and
its fixed nsg=2 / nxpsg=8 geometry may not pay for that — but it is a hypothesis, and
flooring the Proj window at t>=3 is ledgered as needing its own A/B rather than shipped
off this one.

**Fixed in passing: `parity-gate.ts`'s model-process guard false-positived on this
agent harness.** `isModelProcess` parsed every line of `pgrep -fl` output as a record,
and an argv containing embedded newlines (the harness wraps commands as
`zsh -c "cd <repo>\n<command>"`) prints continuation lines with no pid — the fragment
`cd /Users/…/xwen` parsed its second token's basename as a model binary and aborted both
gates. Real records lead with a pid; fragments do not. Unit-tested offline against
captured lines, same as the rest of that matcher.

### Arc 2 — the controller retune, and a harness that makes it repeatable

**Built first: `scripts/retune-draft.ts` (~1280 lines).** The P9 tuning sweep was
hand-driven, and its constants have now been wrong about three successive cost curves,
so the protocol is a script rather than something the next person re-derives. Two stages — a `p_min` grid at margin 1.0, then a margin grid
at stage 1's winner — plus a plain `--no-draft` baseline arm, rep-outermost interleave,
greedy `-n 128` under `XWEN_BENCH=1`, 3 reps, medians. The two P9a tuning prompts moved
to `scripts/lib/draft-prompts.ts` byte-identical and `spec-equivalence.ts` now imports
them, so the two harnesses cannot drift apart. Operational care: a clean child env with
the HF cache root resolved once to an absolute path and `HF_TOKEN` neither passed down
nor serialized, a contention guard, a per-run timeout, and an 0600 raw JSON dump written
atomically after every single run.

**The rule that came out of review: no cell reuse between stages.** The obvious
optimization is to carry stage 1's measurement of the shipped margin into stage 2 and
skip re-running it. That grades two arms from different thermal epochs against each
other, which is the same error the interleaving rule exists to prevent, one level up.
Every stage-2 arm is re-measured. Codex reviewed the harness and raised 3 High, 6 Medium
and 3 Low findings; **all twelve were fixed before any real sweep was scored**, and the
two that would have produced wrong answers were cell-identity rounding collisions and
that stale-epoch stage-2 baseline.

**Results — two independent 120-run sweeps, machine otherwise idle, `lowpowermode 0`
recorded at start and end of each.** Mean-of-medians across both prompts, qualifiers
only (the P9 criterion: ahead of plain's median on BOTH prompts in EVERY rep):

| | run 1 | run 2 | |
|---|---|---|---|
| 27B p_min 0.3 (shipped) | 33.0 tok/s | 33.5 | |
| 27B p_min **0.5** | **37.3** | **37.2** | winner, both runs |
| 27B p_min 0.7 | 36.0 | 36.5 | |
| 35B p_min **0.3** (shipped) | **127.9** | **128.4** | winner, both runs |
| 35B p_min 0.5 | 125.3 | 125.2 | |
| 27B margin 1.0 / 1.2 | 37.5 / 37.5 | 37.8 / 37.8 | a wash |
| 35B margin **1.0** | **129.2** | **128.7** | winner, both runs |

**The 27B wants 0.5 and the 35B wants 0.3, and the mechanism is legible.** At 0.5 the
27B's chat prompt stops pausing entirely — 13-18 paused rounds at 0.2/0.3 down to zero —
and acceptance rises from 57% to 78%, taking that cell from 29.4 to 36.8-36.9 tok/s. The
code prompt already ran pause-free at 0.3 in five of its six reps and gains less
(36.5-37.6 → 37.5-37.9), but at 0.5 it never pauses in any rep. Against
plain that is +46-52%. Pushing on to 0.7 buys 94-95% acceptance and loses tok/s: the
drafts get too short. The 35B is the opposite case — its target forward is cheap enough
that drafting deeper at lower acceptance still pays, and 0.5 would cost it about 2.5%
(125.2-125.3 against 127.9-128.4). So the constant is per-checkpoint, which is what
shipped: `Model::draft_p_min_default()` in `src/hub.rs`, one const arm each.

**`pause_margin` had never actually been swept, and now it has.** P9 validated 1.0 only
against 0.0. Both runs put the 35B's winner at 1.0 with margin 0.8 and 1.2 behind it. On
the 27B at p_min 0.5 the margin is a genuine wash — 1.0 and 1.2 land within 0.1 tok/s in
both runs, and the two runs' nominal winners disagree (1.2 in run 1, 0.0 in run 2, all
inside ~0.5 tok/s of each other) — which is exactly what you expect from a controller
that never pauses at this floor. Margin stays a single shared 1.0, now for a measured
reason.

**The `m=0` diagnostic arm is worth recording, because it is not free.** Margin 0 turns
pausing off entirely, and on the 27B it is simultaneously the single fastest code cell
(medians 39.7 and 40.5, the best numbers in either sweep) and the slowest drafted chat
cell (34.3 and 35.2). With margin > 0 the controller forces a plain round every 32 to keep its
cost EMA fresh; removing those forced rounds shifts the drafter's round alignment enough
to move chat acceptance from 78% down to 73.5%. So the pause machinery costs something
even in a regime where it never pauses, and the two prompt kinds pay and collect
differently — which is why m=0 stays a diagnostic arm and not a candidate.

**Installed.** `draft_p_min` is per-model via `Model::draft_p_min_default()` (0.5 / 0.3);
`DraftArgs.draft_p_min` became `Option<f32>` resolved through it; serve resolves it
through a new `CliOverrides.model_size`; `DEFAULT_DRAFT_P_MIN` is deleted, and
`SpecParams::default()` is documented as a base every real caller overwrites. Two new
tests pin the split (`hub::tests::the_drafting_floor_is_per_checkpoint`,
`serve::config::tests::draft_p_min_defaults_per_checkpoint`, the latter also checking
that a config file naming `p_min` still wins). `cargo test --release` green at 722 + 69.
The script's `SHIPPED_P_MIN` is now a per-size table with a comment saying that
installing a new default means editing both it and `hub.rs`.

**One comparison NOT to make.** The earlier arc's end-to-end figure of 31.7 tok/s for the
27B code prompt at p_min 0.3, and this sweep's own 0.3 arm at 36.5-37.6, come from
different sessions on a machine whose between-session level shifts. That gap is not the coverage
arc's end-to-end gain. The coverage arc's evidence is the verify-round A/B in arc 1; the
retune's evidence is the within-sweep 0.5-vs-0.3 delta. Neither number crosses the
session boundary.

## 2026-08-08 — `mul_mv_ext` ships: the verify forward at span 2 goes 153 → 61 ms, drafted decode gains +11.6-13.2% on the 27B, and the kernel is 20-400x MORE accurate than the mm it replaces

**Context.** The entry below decomposed the verify round's ~149 ms fixed cost and put
87.6% of it in the dense FFN's matmuls at small M — candle's `mul_mm` grid collapsing to
`ne01/64` threadgroups at M ≤ 32, ~73 GB/s against ~280 on the seq==1 mat-vec path. It
named the fix and priced it by byte arithmetic. This arc built and measured it.

**What shipped.** `src/ops/mv_ext.metal` + `mv_ext.rs`: llama.cpp's
`kernel_mul_mv_ext` multi-row quantized mat-vec, which dequantizes a weight block once
and reuses it across 2-5 output rows. Q4_K / Q6_K / Q8_0 × r1ptg 2-5. It is routed at
seq 2..=8 from a **single decision point, `QLinear::forward`**, which is what makes the
blast radius small and legible: that one site covers the 27B dense FFN, the 35B shared
expert, and `forward_all_logits`' lm_head. `XWEN_MV_EXT_CLASSIC` is the kill switch,
`XWEN_MV_EXT_MAX_SEQ` a probe knob, and a `mv_ext` provenance field at schema v8
(grandfather `classic`) records which path a dump ran.

**The gate fixtures never enter the window, so the oracle tests are the correctness
net.** Prefill chunks are 512 tokens and decode is 1, so no parity fixture produces a
2..8 forward at all — the tiers cannot see this kernel. The `mv_ext.rs` oracle tests
against `QTensor::dequantize` at production reduction lengths carry the correctness
claim instead, and `XWEN_MV_EXT_CLASSIC=1` is pinned on both sides of the strict tier
anyway (docs/parity.md "Provenance pins" explains why a kernel the gate cannot exercise
still gets a pin).

**Two-model review, four findings fixed.** Claude byte-diffed the vendored kernel
against ggml and Codex re-derived the dequant arithmetic from scratch; both cleared it.
The fixes were around the edges: `MAX_SEQ` not threaded into the plan, missing Q8_0
ragged coverage, a vacuous offset test, and a wrong geometry claim in the file header.

**The accuracy result runs the other way from `dense_mm`, and it is worth stating
because the reflex is now to expect a precision cost.** This kernel is **20-400x MORE
accurate than the `QMatMul` mm it replaces**: rel_l2 4e-7..8e-6 against ~1.8e-4. The
reason is structural, not tuning — it is f32 end to end where candle's tiled mm stages
weight tiles as half. The oracle tests assert the DIRECTION (`rel <= rel_classic` at
1.0x) rather than an absolute band, so the property is pinned without freezing a number
that a kernel change could legitimately move.

**Both parity gates ALL PASS post-change, reporting pre-change tier numbers** — expected,
since no gate fixture reaches the window, and the confirmation that the schema bump and
the routing change did not disturb anything the gate does see.

**Measured** (round 6; `lowpowermode 0`, both binaries at mtime 20:37, interleaved arms.
Verify bench: 27B, `n_past` 512, 2 reps per arm, means. End-to-end: the P9a protocol —
`spec-equivalence.ts`'s code and chat prompts verbatim, greedy, `-n 128`, `XWEN_BENCH=1`,
drafting default-on, 3 reps, medians).

Verify forward, default against `XWEN_MV_EXT_CLASSIC=1`:

| span | default | classic | |
|---|---|---|---|
| 2 | **61.45 ms** | 153.44 | 0.40x (−92.0) |
| 4 | **85.87** | 176.91 | 0.49x |
| 6 | **125.89** | 197.97 | 0.64x |
| 8 | **161.16** | 220.11 | 0.73x (−59.0) |
| 12-32 | — | — | arms match within 1.2-4.2% (ext inactive) |

One caveat that belongs next to the span-2 number: the default arm's per-rep spread was
large and one-directional — rep 1 faster by 15-30%, the known pattern, the biggest
instance of it seen so far. Bounding the win by the per-rep extremes gives **−87.5 to
−96.4 ms at span 2**, so the sign and the magnitude both survive the spread; the point
estimate is what is uncertain.

**End-to-end drafted decode, the headline:**

| | default | `XWEN_MV_EXT_CLASSIC=1` | |
|---|---|---|---|
| 27B code | **31.7 tok/s** | 28.4 | **+11.6%** |
| 27B chat | **30.9** | 27.3 | **+13.2%** |
| 35B code | **131.1** | 125.8 | +4.2% |
| 35B chat | 119.5 | 119.5 | +0.0% |

The 35B chat cell is a real dead heat, not a measurement failure: it is pause-dominated,
25-26 of 44 rounds, so most of its tokens never enter a verify forward. And the 35B's
small win is what the design predicts — only its shared expert and lm_head route through
`QLinear`, and its verify forward gained just 3.2-4.3 ms at spans 2-8 with nothing
beyond.

**The controller's economics changed, which re-opens the retune.** The 27B default arm
pauses far less than the classic arm — 16 vs 28 rounds on code, 14 vs 32 on chat — and
drafts more. Cheaper verifies make speculation worth attempting more often, exactly as
cheap verifies did after P9a. That is the **third** cost curve `p_min` 0.3 /
`pause_margin` 1.0 have now been wrong about, and the ledger item that was blocked on
this arc is unblocked by it.

**Refuted: extending the window past 8.** `XWEN_MV_EXT_MAX_SEQ=32` makes spans 16 / 24 /
32 **worse than classic by 1.11x / 1.42x / 1.69x**, with span 12 a wash at 0.98x. So
ggml's `ne11_mm_min` 8 envelope is the right ceiling and the inherited default was not
merely untested inheritance — it is now measured. Recorded under decisions.md "Refuted
perf directions"; the ledger item that flagged the upper edge as unmeasured is closed by
it.

**A cross-round caveat, so a future reader does not diff the wrong pair.** The classic
arm on this binary reads ~9-15% slower at mid-spans than round 3's binary did — fixed
intercept 172.9 vs 161.0 ms. Different binaries and machine-state variance; only the
WITHIN-round ratios above are trustworthy, never a cross-round absolute.

**What the fixed cost still holds.** Fitting the spans-2-8 arm leaves ~89 ms of
intercept. Two known non-coverages: the attention projections never touch `QLinear` on
the default path (they are f16 or q8_0 planes), and the single-row lm_head goes through
`forward` rather than `forward_all_logits`. Both are ledgered.

## 2026-08-08 — Verify-round diagnosis: the ~149 ms fixed cost is the dense FFN's matmuls at small M, and none of the armed machinery it was blamed on

**Diagnosis only.** The fix arc is in flight and will write the entry that ships it;
this records what was measured, so the refutations do not have to be re-derived.

**Context.** P9a left the verify round's ~149 ms fixed cost as the new spec-decode
ceiling, with five unpriced candidates: checkpoint materialization, rollback restore,
the trail's host-side conv slices, full-span logits + readback, and command-buffer
syncs. The ledger item's instruction was "price the stages before attacking any of
them". `spec-verify-bench` grew per-stage sync brackets and per-span stack-profile
dumps to do that. Conditions throughout: 27B Q4_K_M, `n_past` 512, `lowpowermode 0`,
medians over 20 reps.

**All five candidates are refuted.** Checkpoint arm costs 5.7 ms fixed — the ~157 MB
of per-round materializes are cheap. Rollback is 2.6 ms fixed + 0.43 ms/tok, and a
keep-4 vs keep-0 branch comparison shows no difference at all. Full-span logits and
readback are 0.12 ms + 0.099 ms/tok, with a last-row-materialize variant reading a flat
~0.4 ms. Against ~149 ms, the entire armed apparatus is a rounding error.

**The cost is inside the verify forward, and it is there unarmed.** Fitting spans 2-32
puts the forward's own fixed cost at ~161 ms, and a span-2 UNARMED forward measures 152
ms against a ~40-43 ms plain step. Speculation does not cause it: a 2-token forward is
just ~3.7x a 1-token forward. Stage decomposition of that span-2 forward against a plain
seq-1 step, both stack-profiled under an identical sync regime, puts **131.8 vs 33.9 ms
in the dense FFN — +97.8 of the +111.7 ms total, 87.6% of it.** lm_head adds +4.4 (3.2x),
mixer_delta +5.9, mixer_full_attn +2.7, and every other bucket is under a millisecond.

**Mechanism: candle's `mul_mm` collapses at small M.** At seq 2..=32 every quantized
matmul takes the tiled path, whose grid degenerates to `ne01/64` threadgroups — ~73 GB/s
effective against ~280 GB/s on the seq==1 mat-vec path. Two refutation rounds confirm
the shape of the problem rather than a mis-set threshold. Forcing the vendored dense
gemm onto small spans (`XWEN_DENSE_MM_MIN_SEQ=1`) moved the fixed intercept only −3.3
ms, because the cooperative-tensor gemm has the same small-M occupancy collapse — though
its marginal did improve, 2.40 → 1.63 ms/tok. And `XWEN_MM_ID_MIN_SEQ=1` on the 35B was
strictly worse at spans 2-8, by +4.1-4.4 ms. No kernel currently in the tree wants these
shapes, so there is nothing to retune toward.

**The fix in flight** is llama.cpp's `mul_mv_ext` multi-row mat-vec: dequantize once and
reuse across 2-5 output rows (ggml-metal-ops.cpp:2120-2223, `ne11_mm_min` 8). Byte
arithmetic says it wins at spans 2-8 and washes by ~16. `src/ops/dispatch.rs:330-334`
already documented this exact gap and pointed at a TODO.md item that did not exist; the
annotation on the fixed-cost item is now that item.

**One consequence worth flagging before the fix lands:** it inverts the `p_min` retune
item. "Longer drafts amortize better" was reasoned off a dominant fixed cost, and
`mul_mv_ext` attacks precisely the short spans, so that tuning conclusion has to be
re-derived rather than carried. Cross-referenced in the ledger, and the retune should
wait.

**Also from these sweeps: the span-48 superlinearity has a suspect.** It is
arming-dependent — every checkpoint-on run overshoots its spans-2-32 extrapolation by
1.54-1.65x while both no-checkpoint runs come in UNDER at 0.80-0.91x — and it does not
move with the dense-mm or mm_id knobs. The profiled armed-minus-unarmed `mixer_delta`
delta grows 6.8 → 160.6 → 304.5 ms at spans 2 / 32 / 48, tracking the K-snapshot plane
buffer (~3.15 MB per plane per layer, ~100-150 MB/layer at spans 32-48). So it is trail
memory pressure, not a kernel threshold. Still outside the production regime and still
unchased. A separate observation from the same sweeps stays unexplained and is NOT
arming-dependent: lm_head roughly doubles at span 48, 7.0 → 13.1 ms, in both armed and
unarmed profiled runs.

## 2026-08-08 — The 27B prefill residual is real and lives in the pipelining: per-stage syncs find only +103 of the +410-438 µs/token, and both cross-chunk accumulation and command-buffer batching are refuted as its mechanism

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

## 2026-07-29 — K-snapshot fused verify lands: spec decode goes from single digits to +8-21%, the 35B flips from -12% to +13-20%, and --draft becomes opt-out

**Context.** P9 shipped speculation as a 27B-only single-digit win with the 35B losing
12%, and its own annotation named the cause: under an armed rollback trail a multi-token
verify chunk fell back to the frozen reference scan, token by token — 39 ms per verified
position on the 27B, so the 48-of-64 DeltaNet layers got no batching win at all inside a
verify forward. TODO.md P9(a) called the K-snapshot fused verify "the precondition for
speculation to pay, not an optimization of it". This arc built it.

**What shipped.** The two fused scan kernels (`kernel_delta_scan`, `kernel_delta_scan_v2`
in `src/ops/delta.metal`) now optionally spill per-token states: a new
`ops::delta_scan_with_trail(.., state_planes)` widens the state output to
`[planes, v_heads, 128, 128]`, most-recent-first (plane s = state after token
`seq-1-s`), mirroring llama.cpp's `kernel_gated_delta_net` K>1 snapshot slots
(ggml-metal.metal:2740-2749) so the CPU oracle stays diffable. Plane 0 is the unchanged
after-loop store — at planes = 1 (every unarmed prefill and decode call) the kernel is
byte-identical to before, proven by a bitwise test. The armed clause is gone from
`linear_attn.rs`'s fused gate; an armed chunk runs the fused scan with planes = seq and
builds the rollback trail from unmaterialized plane views (delta) plus the same
host-side conv-stream slices the reference records. `XWEN_DELTA_CLASSIC=1` still routes
everything, armed chunks included, to `forward_classic` (untouched, still the frozen
oracle). `n_planes` rides the args struct, not a specialized pipeline. Details in
decisions.md "Model math" (the superseded armed-chunk entry).

**Verification.** 700 lib + 66 parity-harness tests green (4 net new; the test encoding
the old fallback was rewritten to assert the new invariant, and the new trail test was
mutation-tested — flipping the kernel's slot mapping fails it while every pre-existing
scan test stays green, so it is not vacuous). Two-model-family adversarial review
(Claude + Codex gpt-5.6-sol at xhigh): zero correctness findings; one doc-accuracy nit
each, both fixed. Both parity gates re-ran and pass with numbers identical to the
pre-change run (35B mm cos 0.999631, Δnll 0.000791; 27B all tiers) — the schema is
untouched.

**Measured** (`lowpowermode 0` — this machine exposes no `powermode` key, so high-power
is never claimed; warm, one model process at a time, interleaved arms, greedy, 128 new
tokens, `p_min` 0.3; two independent end-to-end runs, per-rep values in the raw logs).
The verify A/B is same-day and same-harness: `XWEN_DELTA_CLASSIC=1` IS the pre-P9a
verify path, and that arm reproduced the historical 245 ms @ span ~6 baseline.

- 27B verify round (`n_past` 512): fit over spans 2-32, the marginal cost fell
  **9.42 → 3.57 ms/position** (2.6x) over a fixed cost of ~171 → ~149 ms. At span 6:
  244 → 187 ms/round. In-loop `--stats` corroborate (~248 ms/8.3-position round →
  ~182 ms/7.3).
- 27B end-to-end: code **+19.3 to +21.0%** (29.7-30.0 vs 24.8-24.9 tok/s median), chat
  **+7.6 to +8.4%** — up from +4.8-6.8% / +1.5-7.4%. Acceptance 83.2% / 75.2% (down
  ~4 points from pre-P9a: the batched verify reassociates sums and accepts a slightly
  different token set; throughput improved regardless).
- 35B-A3B end-to-end: code **+18.1 to +19.8%** (124.0-126.7 vs 105.0-105.8), chat
  **+12.6 to +12.8%** — from **-11.5/-12.7%**. The mechanism is the pause controller,
  not the drafter: 35B code went 54-of-66 rounds paused → **0-of-20 paused, 159
  drafted**. The ledger's attribution of the 35B loss to the drafter cache sync (P9b)
  was measured right but read wrong — the ~1.2 ms sync is only fatal when the
  controller pauses and pays it for nothing; cheap verify made it stop pausing.
- spec-equivalence: six of eight comparisons byte-identical; the two 27B-chat forks are
  at the SAME points with the SAME words as the pre-P9a run (the known near-tie class).
  No desync — the sampled stream stays in step for 100+ tokens before forking.

**The `--draft` default flipped to opt-out** — P9(d)'s pre-registered bar ("the 35B at
or above plain, not merely closer") is met with margin on both prompt kinds in both
runs. Zero-flag `generate`/`serve` now load the dflash sidecar; `--no-draft` opts out.

**New open items** (ledgered under P9/TODO): the verify round's ~149 ms fixed cost
(~113 ms above a plain step, now ~60% of a typical round and the new ceiling — price
checkpoint/rollback/readback before attacking); `p_min`/`pause_margin` retune against
the new cost curve (0.3 was fitted to the reference-scan curve); an unexplained
superlinear jump at span 48 in every verify arm (outside the production regime —
block_size 16 caps real spans near 17 — refuted as the dense-mm threshold, cause
unknown); `spec-verify-bench.rs` shipped broken (fixture id `long-swa` predates the P7
rename to `long-mixed` — it had never run in this repo; fixed this arc). The flip's own
review (Codex, clean on the diff) surfaced two default-on consequences, both ledgered:
serve slots persisted without drafter planes now silently decode plain forever while
reporting draft ON (the common hydration path against `--no-draft`-era slots, not the
flag-change edge the code comment was written for), and a custom `--model` GGUF that
fails the drafter preflight now hard-errors at startup where it previously ran plain.

## 2026-07-29 — The 27B prefill gap was the dense FFN's gemm, not the DeltaNet scan: a Q4_K cooperative-tensor kernel takes prefill from 263 to 702 tok/s @925

**Context.** Two entries below, the DeltaNet arc refuted its own premise and handed off a
question: the 27B's 1.8-2.1x prefill loss to llama.cpp is not in the DeltaNet layers, it
is "in the dense projections", and the next arc "should start from a per-stage profile
rather than from a reading of llama.cpp's kernels". A profiling pass did exactly that and
found the answer in one stage. This arc fixed it.

**The profile.** Transcribed from the profiling pass, which produced no docs of its own.
Conditions: `pmset -g` reported `lowpowermode 0` — NOT low-power mode, but this machine
emits no `powermode` key at all, so high-power mode was never positively confirmed and
must not be claimed. `XWEN_BENCH=1` (warm, GPU-complete), GPU exclusivity verified before
every run, fixtures at 880 and 3851 tokens. Rows are tagged MEASURED or DERIVED because
two of the headline figures are derived and one carries a known bias.

Wall (MEASURED, reproduced twice): **2.66-2.68 s @880 (330.7 tok/s)**, **13.86 s @3851
(277.9 tok/s)**.

| stage | @880 ms (% wall) | @3851 ms (% wall) | µs/tok @880 → @3851 | |
|---|---|---|---|---|
| Dense FFN (64 layers) | 2268 (85.3%) | 9926 (71.6%) | 2578 → 2578 | DERIVED |
| DeltaNet non-scan (48) | 278 (10.4%) | 1216 (8.8%) | 316 → 316 | DERIVED |
| Attention (16) | 155 (5.8%) | 914 (6.6%) | 176 → 237 | MEASURED |
| DeltaNet scan (48) | 136 (5.1%) | 377 (2.7%) | 154 → 98 | MEASURED/DERIVED |
| Prefill mask (hoisted) | 4.01 (0.2%) | 51.22 (0.37%) | 5 → 13 | MEASURED |
| **sum** | **2841 (106.8%)** | **12484 (90.1%)** | 3229 → 3242 | |
| unaccounted | −181 (−6.8%) | +1376 (+9.9%) | −206 → +357 | |

**Read the −181 ms.** A negative residual is physically impossible: the stage sum exceeds
the wall clock. The cause is that the FFN row is DERIVED from an isolated T=512 rate that
is ~7-8% pessimistic against what the FFN achieves inside a real forward. So the honest
claim is a **band: the dense FFN is 66-85% of prefill wall time**, and the two DERIVED
rows carry that bias. Everything else is measured directly.

**Two protocol traps the profiling pass documented, both easy to repeat.** (1) All rates
above are AMORTIZED — BATCH=8 dispatches per sync, outputs held alive so the allocator
pool cannot inject a false write-after-write barrier. Per-dispatch numbers charge a
command-buffer round trip a real forward never pays, and are wildly different (attention
102.12 vs 57.13 ms/layer, DeltaNet 8.903 vs 3.367, FFN 24.589 vs 20.620); building the
budget from them sums to **127% of wall**, i.e. nonsense. (2) These benches throttle: the
same T=3851 q4_K gate measured 53.773 ms in a 9.3 s run, 54.118 ms in a low-duty run, and
66.639 ms (23% slower) in a 36 s run. Low-duty-cycle only; a profiling run long enough to
heat the GPU inverts conclusions.

**The mechanism, and it is not subtle.** The FFN runs `DenseMlp` → `QLinear::forward` →
candle `QMatMul` → `kernel_mul_mm_q4_K_f32`; the same shapes through `ops::matmul_f16` →
`kernel_mul_mm_f16_f32_t` are 2.4-3.0x faster:

| [T, 5120] × [5120, 17408] | Q4_K (`QMatMul`) | f16 cooperative-tensor gemm |
|---|---|---|
| T = 512 | 7.893 ms (11.6 TFLOP/s) | 3.242 ms (28.2 TFLOP/s) |
| T = 880 | 13.143 ms (11.9) | 4.412 ms (35.6) |
| T = 3851 | 54.118 ms (12.7) | 19.979 ms (34.4) |

(Profiling pass's figures. Re-measured independently for this entry with
`dense_ffn_prefill_timing`: 8.170 / 2.988 at T=512, 12.390 / 4.188 at 880, 55.904 /
19.932 at 3851 — same conclusion, within 8% on every cell.) The f16 rate is not a
synthetic-shape artifact: the DeltaNet projections, which DO take the f16 plane in
production, measure 27.7-30.4 TFLOP/s at T=880. The reason the FFN does not is that the
dual-plane machinery (`Weights::attn_proj`, gguf.rs:657) has exactly one caller,
`Proj::load` in attention.rs:108.

**Why this is kernel efficiency and not a memory wall — stated without appeal to a peak
bandwidth nobody measured.** At T=512 the Q4_K arm reads 50.14 MB of weights and the f16
arm 178.32 MB. The Q4_K arm moves **3.6x fewer weight bytes and takes 2.4x longer**
(10.9 vs 66.0 GB/s of weights+output). If either arm were bandwidth-bound, the one moving
fewer bytes would be the faster one. It is not.

Counterfactual (DERIVED, and an **upper bound** — it assumes a quantized gemm could reach
the full f16-plane rate, which is not guaranteed): 694 tok/s @880 and 496 @3851 against
llama.cpp's 486 / 502.

The same profile refuted a standing suspect and left two others open. All three are now
TODO.md ledger items rather than folklore. Their figures are the profiling pass's, carried
over rather than re-measured here — with one exception, flagged where it appears: the
attention-glue split came from the arc's briefing, not from the raw profile.

**Refuted: the materialized causal mask.** `flash.metal` is genuinely unreachable at head
dim 256 (`ops::flash_attn` hard-bails at `head_dim != 128`, dispatch.rs:3324/3361, zero
production callers), so prefill runs candle sdpa against a host-built mask — but that
mask is **hoisted**: built once per chunk in `model.rs` `run_stack` and shared across all
16 full-attention layers, NOT rebuilt per layer. The profiling pass's own first run got
this wrong and multiplied by 16, turning a 51 ms non-event into a ~682 ms scare; the
corrected figure is **0.37% of wall at 3851** (51.22 ms) and 0.15% at 880. Mask + sdpa
together grow ~1.2 percentage points between the two lengths against an observed 16%
throughput drop — an order of magnitude short of what the hypothesis needs. The 402 MB is
DERIVED (Σ over chunks of `n_head × seq × k_seq × 2`), not measured.

**Open: a length-dependent residual outside every measured stage.** Per-token wall goes
3023 → 3599 µs between the two fixtures (+576, the 330.7 → 277.9 drop), and the four big
stages account for only **+13 µs/token** of it — the FFN and DeltaNet non-scan rates are
flat, and the scan actually gets *faster* per token with length. But part of the residual
swing is an artifact of the DERIVED FFN row's 7-8% bias, so the defensible statement is
**+350 to +560 µs/token**, not a hard +576. Not attributable without per-layer
instrumentation inside `run_stack`.

**Open: attention glue.** Attention is 57.13 ms/layer amortized at T=3851 (up from 5.15
at a single 512-token chunk at position 0, growing monotonically with position), and the
brief that opened this arc put ~42.43 ms/layer of that in ~10 unfused eager passes —
a figure that appeared in the briefing rather than in the raw profile, so treat it as
indicative until re-measured. `ops::attn_gate` already exists but is wired only into the
DFlash path.

**Built: `src/ops/dense_mm.metal`.** A dense Metal-4 cooperative-tensor gemm that reads
Q4_K directly and dequantizes each weight tile in registers. It is `f16_t.metal`'s kernel
with exactly one substitution — the A-tile staging phase gains ggml's block-quant
dequant instead of a half widen-copy — so the 64-row × 128-token tiles, the f32
activation read straight from device as a cooperative tensor with no B staging, the
reduced-precision matmul2d descriptor and the extent-clipped store all carry over
unchanged. That is also how ggml writes it: templated over
`(block_q, nl, dequantize_func)`. Instantiated for q8_0/q4_K/q5_K/q6_K; q4_K is the
production one.

**The alternative was priced, not hand-waved.** Dequantizing to a transient f16 scratch
and feeding the existing gemm needs 178 MB written and 178 MB read back per projection
per chunk against the 50 MB of Q4_K the kernel otherwise streams — ~8x the weight
traffic, ~0.68 ms per projection at 600 GB/s against the gemm's own 3.2 ms, so about a
fifth of the win handed straight back plus a 178 MB scratch buffer. Permanent f16 planes,
the trick the attention projections use, are not even arithmetically available here:
17.1e9 FFN parameters at 2 bytes is 34 GB. The planes that did ship cost nothing —
`Weights::qlinear_with_plane` reuses `qlinear_with_buffer`'s one-upload construction, so
the `QLinear` decode path and the prefill gemm index the same allocation.

**Threshold from the tile geometry, not from tuning.** `DENSE_MM_MIN_SEQ = 32`,
exclusive (`seq > 32`), following `F16_MM_MIN_SEQ` and ggml's `ne11_mm_min`. candle tiles
tokens 32 wide and the vendored kernel 128, so up to 32 tokens both sit on the same
launch-latency floor — 1.01-1.05x, a wash — and at 33 candle takes a second token tile
while the vendored gemm does not. Isolation, plateau ms, interleaved arms:

| tokens | 33 | 48 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| [17408, 5120] speedup | 1.48x | 1.30x | 1.94x | 2.19x | **2.41x** |
| [5120, 17408] speedup | 1.20x | 1.19x | 1.65x | 2.71x | **3.02x** |

Production prefill runs in 512-token chunks (`PREFILL_CHUNK`), so the kernel always sees
the end of that curve.

**End to end** (27B, `lowpowermode 0` — not LPM, high-power mode not separately
confirmable on this machine — `XWEN_BENCH=1`, `--no-draft`, committed fixtures,
interleaved arms F/C/F/C, 5 reps, medians with ranges; the classic arm doubles as the
calibration — at 263 tok/s @925 it reproduces the documented ~270 baseline, which is how
this run was distinguished from three earlier contended ones that read 3x low in **both**
arms. Note the profiling pass's own baseline on a quieter machine was higher still,
330.7 tok/s @880, so the classic column here is a valid control but not a machine-best
number):

| | fused | classic (`XWEN_DENSE_MM_CLASSIC=1`) | |
|---|---|---|---|
| prefill @925 | **702.2** [637-715] | 263.1 [230-307] | **2.67x** |
| prefill @3851 | **444.9** [409-450] | 199.9 [194-205] | **2.23x** |
| decode n=128 | 24.6 [20.6-24.7] | 22.1 [19.5-23.4] | no regression |

Decode cannot change and did not: at seq 1 the threshold sends it down the `QLinear`
chain, and `dense_mlp_below_threshold_takes_the_classic_path` asserts bit-identical
output there against a block built with no planes at all. The 1.11x in that row is
between-run drift, not an effect — the two ranges overlap almost completely. The 35B-A3B
is untouched by construction (every layer is MoE; `DenseMlp` never constructs) and
spot-checks at 1.01x prefill / 0.94x decode, both noise.

So 27B prefill goes from 1.8-2.1x behind llama.cpp to ahead of it at 925 (702 vs 486)
and near parity at 4k (445 vs 502). Against the counterfactual — which was an upper
bound, assuming a quantized gemm could reach the full f16-plane rate — the short prompt
landed on it (702 vs 694 predicted) and the long one came in under (445 vs 496). The
shortfall at 4k is where the remaining work is: the length-dependent residual above is
now a much larger share of what is left than it was of what it came out of.

**The cost, stated plainly.** This kernel is **less accurate than the `QMatMul` chain it
replaces**. Against a dequantize-then-f32 oracle at the 27B FFN shapes it lands ~4.1e-4
rel_l2 where candle's kernel lands ~1.9e-4; the two differ from each other by ~3.7e-4.
Both stage the weight tile as half and accumulate in f32 — the extra ~2e-4 is matmul2d's
reduced-precision tensor-core path, which is where the throughput comes from. It is not a
new precision class (the attention prefill gemm made the identical trade, and llama.cpp
sets the same descriptor flag for its own dense FFN), but it is a real one, so the kernel
is graded like the DeltaNet scan and not like the glue kernels:
`XWEN_DENSE_MM_CLASSIC=1` is pinned on **both** sides of the strict tier, a `dense_mm`
provenance field (parity_schema v7, grandfather `classic`) proves which side ran what,
and the bounded mm/decode/ppl tiers carry the signal against frozen floors. Cached pre-v7
references stay valid — no binary of that era had the gemm.

**Both gates pass on frozen floors.** 27B: strict `cos=1.000000`, mm `cos=1.000000`,
decode 64/64 with 0 mismatches on all three fixtures, ppl `Δnll=0.000243`. 35B-A3B
(mandatory here because the provenance schema moved to v7, not because any 35B code
path changed): strict `1.000000`, mm `0.999631`, decode 63/63/62 with 0 mismatches, ppl
`Δnll=0.000791` — reproducing the DeltaNet entry's numbers exactly, which is the
confirmation that the 35B is bit-for-bit untouched.

**Shipped.** `src/ops/dense_mm.metal` + `src/ops/dense_mm.rs` (new library, compiled
lazily so nothing else gains a Metal-4 dependency), `dispatch::run_matmul_dense_q_mm` and
its dtype/geometry support matrix, `gguf::QuantPlane` + `Weights::qlinear_with_plane`,
`DenseMlp`'s prefill entry, and the `XWEN_DENSE_MM_CLASSIC` / `XWEN_DENSE_MM_MIN_SEQ`
switches wired into `parity-gate.ts`'s strip list, `referenceEnv()`, the strict candidate
env and `isReferenceDump()`. New tests: kernel-vs-oracle and kernel-vs-`QMatMul` at the
production FFN shapes, all four dtypes, the tile-edge cases (token counts across the
128-wide tile, out-dims across the 64-wide one), the instantiation-matrix and
geometry-vs-`#define` cross-checks, a `DenseMlp`-level fused-vs-classic comparison with
per-projection attribution, and the threshold-gating test above. 696 lib tests and 66
parity tests pass, 0 failures.

## 2026-07-29 — DFlash adapted to the Qwen sidecars (P9): both drafters load and accept 85-95%, and speculation is a 27B-only win because the verify forward runs the per-token reference scan

**Context.** The DFlash subsystem came over from laguna whole and inert: `dflash.rs`
described a six-layer, hidden-3072, 72-head drafter with a per-head softplus attention
gate and a per-tap encoder norm, gated behind a `dflash.decoder_arch == "laguna"` check.
The ggml-org sidecars for Qwen 3.6 are a different model — 5 layers at hidden 5120 (27B)
or 6 at 2048 (35B-A3B), 32 Q / 8 KV heads, no gate tensor, no `enc.aux_norm`, no
`decoder_arch` key at all, and sliding-window attention on every layer but the last. Two
tests (`real_file_load_and_shapes`, `real_file_bf16_alias_load_and_forward`) were the
suite's only red ones and asserted the laguna geometry on purpose, as the arc's gate.

**What the oracle actually says.** Three of the graph's differences were not in the
briefed list and were found by reading `reference/llama.cpp/src/models/dflash.cpp`
against the shipped headers:

- **The noise block is non-causal.** `common/speculative.cpp:1004` calls
  `llama_set_causal_attn(ctx_dft, false)`, and the causal branch of the KV mask builder
  is guarded by `if (causal)` (llama-kv-cache.cpp:1793). The drafter is a block-diffusion
  model: it denoises `[id_last, MASK × 15]` in one forward, so every block position sees
  every other in both directions. The inherited code masked the block causally.
- **The injection path applies no `attn_norm`.** dflash.cpp:252-253 projects the raw
  encoder output into `wk`/`wv`; `enc.output_norm` is the only norm on that path. The
  query path in `draft_forward` does apply `attn_norm`, so the two deliberately disagree.
  The inherited code normed both, citing the laguna branch — which no longer exists.
- **The encoder is three ops.** dflash.cpp:109-123: concatenate the taps tap-major,
  `fc`, `enc.output_norm`. No per-tap RMS-norm, no per-tap scale.

Everything else the mapping pass had established held up: taps are the residual
*entering* the named target block, so the `t - 1` translation to our `l_out` capture
points is unchanged; rope is plain NEoX over all 128 dims (no `rope.dimension_count` key
→ `n_rot = n_embd_head_k`), theta 1e7 from the GGUF; QK-norm before rope; SwiGLU with a
real `ffn_norm` tensor.

**Sliding windows, implemented as a narrow rather than a mask.** llama.cpp masks
`p1 - p0 >= n_swa`, i.e. a query at position `p` keeps `[p - window + 1, p]` on the past
side and — being non-causal — everything on the future side. The obvious implementation
is an additive mask over the full score row, but the block's windows are a contiguous
union: query 0 has the deepest floor, and the 16 queries' floors span at most 15
positions. `attention` therefore narrows the cache to `[lo, committed + n_block)` and
masks only the ≤15 columns between the individual floors, falling back to no mask at all
while the context still fits inside the window. A windowed layer costs O(window) per
round instead of O(context), which retires half of the argument behind
`DEFAULT_DRAFT_CTX` being small: only the final full-attention layer — one of five or six
— still grows with depth.

**Measurements** (`lowpowermode 0`, warm, greedy, 128 decoded tokens, interleaved arms
within one process-per-run sweep, 3 reps, medians; scripts in the session scratchpad,
`scripts/spec-equivalence.ts` committed):

| | plain | `--draft` (p_min 0.3) | acceptance |
|---|---|---|---|
| 27B, code prompt | 25.0-25.2 tok/s | 26.4-26.7 (+4.8 to +6.8%) | 87.4% |
| 27B, chat prompt | 20.6-21.5 tok/s | 20.9-23.1 (+1.5 to +7.4%) | 65.9-73.7% |
| 35B-A3B, code prompt | 105.1 tok/s | 93.0 (-11.5%) | 81.3% |
| 35B-A3B, chat prompt | 105.5 tok/s | 92.1 (-12.7%) | 82.8% |

The 27B rows are ranges across two independent interleaved runs rather than one run's
median, because that model's run-to-run spread is wide enough to matter (docs/decisions.md
"Measurement discipline", and the 27B caveat under TODO P11): within a run the reps are
tight — 26.9/26.7/26.7 against 24.8/25/25 on the code prompt — but the level shifts
between runs. The sign is stable; the magnitude is ±2 points. The 35B rows repeated to
within 1%.

The drafter proposes well on both models — 85-95% acceptance at `p_min` 0.5-0.9, and a
27B run at `p_min` 0.9 accepted 54 of 54. The 35B's loss is not a drafting failure.

**The 35B loses ~12% before it drafts anything.** An arm with `--draft-p-min 1.1`, where
no drafter token can ever clear the threshold and 119 of 127 rounds pause, still decodes
at 92.6-92.7 tok/s against 105.1-105.5 plain — the same loss as the best drafting arm.
The cost is the drafter's per-round cache sync: every committed token runs `encode` (an
8-tap concat through a [2048, 16384] `fc`) plus six layers of `wk`/`wv` projections,
QK-norm, rope and two `slice_set`s, about 14 small Metal dispatches for ~1.2 ms. That is
12% of a 9.5 ms plain step and 2.8% of the 27B's 43 ms one, which is the whole difference
between the two models' verdicts. It is dispatch-bound, not FLOP-bound — the same disease
the MoE glue fusion cured — and it is mandatory while a drafter is attached, because a
drafter whose cache falls out of step with the target's can never resume speculating.

**The verify forward gets almost no batching win, which is the real ceiling.** Under an
armed rollback trail a multi-token chunk falls back to the frozen reference scan
(linear_attn.rs:194-205), and that scan walks tokens one at a time in candle ops. So the
layers that are 48 of the 27B's 64 and 30 of the 35B's 40 cost the same per position in a
16-token verify as in 16 single-token steps: measured 245 ms for a ~6-position verify on
the 27B against a 43 ms plain step, i.e. 39 ms per verified position. Speculative decoding
is a bet that verifying N tokens costs far less than decoding them one at a time; on this
architecture that bet currently pays only in the attention and FFN layers. **The
K-snapshot fused verify is therefore not an optimization of P9 but the precondition for
it** — TODO.md P9 carries it with the structural note that both scan kernels already hold
each thread's state slice in registers across the timestep loop, so emitting per-token
snapshots is one guarded store plus a wider output buffer.

**Tuning.** `draft_p_min` swept over {0.2, 0.3, 0.5, 0.7, 0.9} on the 27B: 0.3 is the only
value that came out ahead of plain on both prompt kinds in every run (0.5 lost on the chat
prompt twice), so the default moves 0.5 → 0.3 in `SpecParams`, the CLI and the serve
config. Lower `p_min` drafts longer at lower acceptance, and that wins precisely because
the verify cost is near-linear in its span — with no batching penalty to pay, a longer
span amortizes the round's fixed cost. That reasoning is fitted to the reference-scan cost
curve and should be re-run when the fused verify lands. `pause_margin` stays 1.0: the
controller earns it on the 27B, where always-drafting (`--draft-pause-margin 0`) measured
21.8 tok/s against the controller's 25.8 and plain's 23.3. It cannot help the 35B, whose
loss is charged to rounds the controller has already paused.

**`--draft` stays opt-in.** The flip to opt-out was conditional on the controller holding
a never-materially-slower property on both models. It does not: no setting of `p_min` or
`pause_margin` can recover a 12% loss that is incurred on paused rounds. The CLI, serve
config and `--init` template text all lose their "not adapted yet, fails at load" wording
and gain the measured reason instead.

**Equivalence, in two modes.** `scripts/spec-equivalence.ts` diffs `--draft` against
`--no-draft`. Greedy mode (temperature 0) checks the verify walk's token selection; it
found 11 of 12 comparisons byte-identical, the twelfth forking on the 27B chat prompt at an
adjective ("accessible" vs "educational"). That is the batched verify forward reassociating
its f32 sums differently from the single-token forward and flipping a near-tie — the same
class the decode parity tier's near-tie rule grades, and the same one already recorded for
the fused delta scan under P8a.

Greedy mode has a structural blind spot, though, raised by the second-family review: at
temperature 0 the argmax path never draws from the seeded RNG, so no amount of greedy
agreement can show that the spec loop advances the SAMPLER STREAM the same number of times
the plain loop does — one extra or missing draw would reroute every subsequent token. A
`sampled` mode now covers it: temperature 0.8 at a fixed seed with `--draft-p-min 0` and
`--draft-pause-margin 0`, so every round drafts a full block and nothing pauses (auto-pause
is what makes a temperature>0 run irreproducible from a seed, so it has to be off).
Result: the 35B is byte-identical on both prompts with 360 and 435 drafted tokens over 384
and 464 verified positions, and the 27B is identical on the code prompt with 315 drafted.
**The sampler stream is in lockstep.** The 27B chat prompt forks in sampled mode too, at
line 12 — deep enough that the stream was demonstrably in step for over a hundred tokens,
which is the near-tie signature rather than the desync one (a desync reroutes the first
sampled token, and the script says so when a sampled-mode fork lands on line 1).

Both modes now also refuse to report OK on a run that drafted nothing: a comparison that
paused into plain decoding exercises no verify and no rollback, so its agreement means
nothing, and the script says NO COVERAGE instead. It rebuilds before comparing and checks
the binary is not older than the newest source, so it cannot bless a stale build.

**Deleted, added, changed.** Deleted: the `decoder_arch` requirement, the `enc.aux_norm`
tensor and its per-tap norm-and-scale, the `attn_gate` tensor and its softplus output
gate, the `softplus` helper, and the within-block causal mask. Added: `sliding_window` +
`swa_layers` on `DflashConfig` with a `layer_window(il)` accessor, `value_bool`, the
narrow-plus-mask windowed attention, `Model::draft_kv_bytes_per_token()` (40 KiB/token on
the 27B's five layers, 48 on the 35B's six — `serve/config.rs`'s hardcoded 35B-only
constant now derives from it), and `scripts/spec-equivalence.ts`. `attach_drafter` also
gained the mismatched-pairing check the CLI path never needed while every drafter load
failed: nothing in a sidecar's metadata names its target, so `--draft <path>` can pair the
27B drafter with the 35B, which used to reach `set_spec_taps`'s `assert!` and panic. It is
now an error naming both numbers, and the hidden-size mismatch is caught at attach rather
than at the first forward. Review caught that serve had the same hole one level earlier:
`check_draft_geometry` compared head counts but not hidden size, and the tap check alone
does not separate the two sidecars in one direction — the 35B-A3B drafter's translated
taps top out at 37, inside the 27B's 64 layers — so `xwen serve --model-size 27b --draft
<35B drafter>` passed startup validation and failed the first job instead. The check is
now threaded through `read_draft_config` from the target's `XwenConfig`, which is what
`validate_model` exists to do. Tests: the two red ones
rewritten against real sidecar geometry and parameterized over both models (the alias test
now injects 2100 positions, past the 27B's 2048 window, so the narrow-plus-mask path runs
on real weights), plus three new ones — a windowed forward graded against the scalar
reference and against an unwindowed twin, a perturbation test proving the last block row
informs the first, and a config test for the window keys. `ops/bf16.rs`'s `DRAFTER_SHAPES`
covers both sidecars' twelve production matmul shapes. Suite: **760 passing, 0 failing**,
the two deliberately-red tests among them.

## 2026-07-29 — The DeltaNet scan is 3% of 27B prefill: llama.cpp's decomposition measured slower, and the premise behind P8b refuted

**Context.** The head-to-head entry below named the sequential DeltaNet scan as the cause
of the 27B's 1.8-2.1x prefill loss, and a mapping pass then established that llama.cpp's
Metal path does NOT run the chunked form it advertises — its fused
`ggml_gated_delta_net` op pre-empts the chunked graph (delta-net-base.cpp:437-446), so it
runs the same sequential scan we do, under a far more parallel decomposition. That made
re-decomposing our kernel the obvious lever and demoted P8b's chunked scan. This arc
built the re-decomposition. It lost, and finding out why cost the premise as well.

**Built.** `kernel_delta_scan_v2` is llama.cpp's shape adapted to our layouts: one
SIMDGROUP owns one state value-column for the whole T loop, both key-dim contractions
collapse to `simd_sum`, no barrier appears anywhere in the timestep loop, and the grid is
1536 threadgroups at the 27B geometry against the shipped kernel's 192. It needs q and k
pre-normalized, so `kernel_delta_l2norm` hoists the L2 clamp-norm out into its own
dispatch. Both are bounded against the frozen reference scan exactly as the shipped
kernel is, at the same tolerances — no parity schema change, floors untouched.

**Measured, and it lost.** Isolation timing per DeltaNet layer (`delta_scan_timing`,
plateau ms, each arm inclusive of its q/k norm), and interleaved end-to-end (median of 3
rounds, arm order flipped each round, full power `lowpowermode 0` start and end,
`XWEN_BENCH=1`, `--no-draft`, committed fixtures):

| isolation, one layer | 27B @880 | 27B @4096 | 35B @880 | 35B @4096 |
|---|---|---|---|---|
| shipped | 1.97 | 8.56 | 1.57 | 6.31 |
| llama.cpp shape | 2.73 | 14.81 | 1.88 | 8.93 |

| end to end, tok/s | 27B shipped | 27B v2 | 35B shipped | 35B v2 |
|---|---|---|---|---|
| prefill @880 | 296.9 | 307.6 | 2504.9 | 2497.2 |
| prefill @3851 | 262.1 | 257.7 | 2312.3 | 2398.6 |
| decode n=256 @630 | 22.0 | 22.1 | 103.3 | 102.2 |

The end-to-end arms are a tie in both directions — which is the actual finding, not a
measurement failure. **The scan is 3% of 27B prefill.** 48 layers × 1.97 ms is 95 ms of a
2.96 s prefill at 880 tokens; × 8.56 ms is 411 ms of 14.2 s at 3851. A free scan buys
~297 → ~307 tok/s against llama.cpp's 486. The 1.7x isolation regression at 4096 shows up
end-to-end as exactly the 1.7% it should, which is the cross-check that the share is
right. The mechanism of the loss is q/k read amplification: our threadgroup stages one
normalized q and k vector per threadgroup per timestep, llama.cpp's shape gives every
simdgroup its own copy — 32x the L2 traffic — and at head dim 128 each lane owns 4 state
entries, so two `simd_sum`s ride on 8 useful FMAs. Hoisting the norm alone loses too: the
extra dispatch rewrites the whole q|k plane (0.52 / 1.80 ms) to save a reduction already
amortized over a threadgroup.

**Shipped: nothing.** The scan kernel is byte-identical to what it was before this arc,
which both gates confirm by reproducing every number exactly (35B strict 1.000000, mm
0.999631, decode 63/63/62 with 0 mismatches, ppl Δnll 0.000791; 27B strict and mm both
1.000000, decode 64/64/64 with 0 mismatches, ppl Δnll 0.000330). `parity-gate.ts`'s
`baseEnv()` strips `XWEN_DELTA_SCAN_V2` alongside the other presence-based switches —
unlike its siblings it has no provenance field, so a stray shell value would have applied
to both sides and passed the gate while grading the wrong kernel. The two new kernels stay
behind `XWEN_DELTA_SCAN_V2=1` as a runnable refutation — the kernel they mirror is
vendored in this repo and will invite the same proposal again — along with
`ops::delta_l2norm`, its bounded test, and the `delta_scan_timing` bench that produced
the table. 681 lib tests pass with the two dflash `real_file` tests staying deliberately
red, 63 parity tests pass (3 ignored, the ones the gate feeds), and the delta and
linear-attn suites pass under `XWEN_DELTA_SCAN_V2=1` as well as under the default.
`scan_matches_reference` picked up seq 67 — prime, and a multiple of no tile or simd
width the scan is built from.

**What this hands off.** The 27B prefill gap is still ~1.8-2.1x and is now known NOT to
be in the DeltaNet layers. It is in the dense projections; that is the next arc's
question, and it should start from a per-stage profile rather than from a reading of
llama.cpp's kernels. P8b's chunked scan keeps its rollback-replay rationale and loses its
prefill bounty (see TODO.md and decisions.md, "The DeltaNet scan decomposition").

## 2026-07-29 — Two-family review of the MoE-glue + top-p diff: kernels clean twice over, one real hole in the gate script

**Context.** Standard post-arc review of what became commit `bec5fa2`, one Claude
reviewer and one outside-model pass (Codex CLI, gpt-5.6-sol, xhigh effort) over the full
diff. Both independently cleared the fused kernels line by line — barriers, the bitonic
network's pad ordering, the epilogue's fp pragmas, wrapper validation. The findings that
survived were all at the edges.

**Fixed.** (1) `parity-gate.ts`'s `baseEnv()` stripped every presence-based kernel
switch EXCEPT the two new ones, so a stray `XWEN_MOE_GLUE_CLASSIC` in the shell (even
`=0`) would have silently classic-ed both sides and passed the gate without dispatching
either new kernel. Both are stripped now, and the 35B gate was re-run under an
explicitly clean env: ALL PASS, every number identical (strict 1.000000, mm 0.999631,
decode 63/63/62 with 0 mismatches, ppl Δnll 0.000791) — which also retroactively
validates the pre-fix run. (2) `top_p == 0.0` — reachable through the serve layers —
had no test; it now pins the keep-one behavior against the llama.cpp oracle.

**Documented, not changed.** The top-p cut is one ulp off llama.cpp at exact f32
boundaries (they re-softmax survivor logits, we divide full-softmax survivors by their
sum — algebraically equal, reproduced counterexample in decisions.md), and the
`llamacpp_filtered` oracle is structurally blind to that class. **Ledgered.** `top_k=0`
semantics (greedy here, disabled there), the unpinned
`MTLMathFloatingPointFunctions` compile axis under the epilogue's bare `exp` (the
bitwise suites are the tripwire), and `mul_mv_id_dual`'s trusting wrapper. Codex's one
discounted finding — the ragged q4_K tail overread — is the documented, deliberate
ggml-matching inheritance, predating this arc.

## 2026-07-29 — First llama.cpp head-to-head: xwen wins decode on both models, loses 27B prefill 2x to the sequential DeltaNet scan

**Context.** Three perf arcs (sampler tail, fused DeltaNet, fused MoE glue) had moved
xwen's own numbers, but nobody had ever measured llama.cpp on this machine. Same GGUF
files fed to both engines (the blessed Q4_K_M pair), llama.cpp at the pinned oracle
build e9fa078 with `-fa 1`, strictly interleaved arms per the measurement-discipline
entry, 6-7 reps per cell, decode at matched context depth 630. Power mode: AC,
`lowpowermode 0`, no `highpowermode` key — Automatic, the whole run.

**Results** (median tok/s, ratio = xwen/llama.cpp):

| cell | llama.cpp | xwen | ratio |
|---|---|---|---|
| 35B decode tg256@630 | 98.0 | 103.1 | 1.05 |
| 35B prefill @925 / @4096 | 2725.7 / 2497.9 | 2546.6 / 2316.6 | 0.93 / 0.93 |
| 27B decode tg256@630 | 19.2 | 19.6 | 1.02 |
| 27B prefill @925 / @4096 | 486.2 / 501.8 | 268.9 / 235.6 | 0.55 / 0.47 |

**Reading.** Decode is won on both models — the design target metric. The 35B prefill
deficit is mostly thermal-boost asymmetry: llama.cpp's early reps boost to ~3140 and
settle to ~2500-2700 (-17%) while xwen only drifts -5%; restricted to settled reps the
ratios are 0.96 @925 and 0.90 @4k, i.e. near parity at 925. The 27B prefill gap is
real, large, and length-growing: xwen *degrades* 269→236 from 925 to 4k while llama.cpp
*improves* 486→502. The cause is the one CLAUDE.md's P8 note predicted — the 27B runs
48 sequential DeltaNet reference scans at inner width 6144 (vs the 35B's 30 at 4096),
and llama.cpp prefills those layers with its chunked (chunk=64) form. P8b (chunked
scan) is the named fix and now has a measured bounty: ~2x on 27B prefill.

**Sampler asymmetry checked.** llama-bench's tg loop runs no sampler; xwen's decode
number carries full 1.0/0.95/20 sampling. Greedy control run: 104.9 vs 103.1 — the
sampler costs ~1.5%, so the decode win is not an artifact of what each side skips.

**Also confirmed.** The 2026-07-29 MoE-glue numbers reproduce independently (35B decode
103.1 vs 102.8 quoted, 27B 19.6 vs 19.0). Raw per-rep data and harness scripts are in
the session scratchpad (results-{35b,27b}.txt), not the repo.

## 2026-07-29 — Top-p switched to llama.cpp's renormalizing convention

**Context.** The 2026-07-28 sampler rewrite kept candle's top-p rule — cut against
full-vocabulary mass, skip the cut entirely when the top-k set holds less than `top_p`
of the total — and ledgered the divergence rather than fixing it inside a perf change.
llama.cpp is this project's ground truth for everything else, and `--top-p 0.95` did not
mean what a llama.cpp user would read it as.

**Change.** `truncate_top_p` is now `llama_sampler_top_p_apply`. `top_p >= 1.0` returns
without touching anything (llama.cpp builds an empty sampler there); otherwise the top-k
survivors are renormalized to sum to one and the walk keeps the shortest prefix whose
cumulative mass reaches `top_p`, comparing `cum_sum >= top_p` with the crossing token
INCLUDED. `min_keep` is not carried — llama.cpp's default is 0, disabled, and the loop's
own at-least-one guarantee is all that default gives. The two branches of
`candidate_set` collapsed into one: the "is the top-k mass already under `top_p`" guard
existed only to express the absolute-mass rule and has no analogue here. The
renormalization is CPU-side over the ≤20 candidates and costs nothing; the device-side
full-vocabulary softmax stays for now.

**What this changes.** Sampled outputs, one-directionally: renormalizing cuts the same
or more, never less, so a seeded stochastic run draws a narrower candidate set than a
pre-2026-07-29 build wherever the cut bites. Accepted. Greedy is untouched and the
parity gate is greedy end to end, so nothing there moves. A side effect worth naming:
the cut no longer compares absolute mass, so the softmax denominator divides back out of
it, and the device fast path and the CPU `SampleControl` path can no longer truncate
differently by an ulp at the threshold.

**Tests.** `top_p_measures_absolute_mass_not_renormalized_mass` is replaced by
`top_p_measures_mass_renormalized_over_the_candidate_set`, built on two hand-computed
rows where the conventions give different answers — one where the old rule's cut ran and
stopped short, one where it was skipped outright because the top-k set held only 0.76 of
the mass. A `llamacpp_filtered` transcription joins the candle transcription as an
oracle, and the candle-equivalence matrix now claims candle agreement only at `top_p`
1.0, where no cut applies; below that it is llama.cpp the candidate set is checked
against, at every shape and every p. 29 sampler tests pass, clippy clean.

**Not done.** The perf half of the ledger item — the fast path no longer needs a
full-vocabulary softmax, since renormalizing over k survivors IS a k-wide softmax — is
split out as its own TODO entry. It is gated on a Metal top-k, not on this change.

## 2026-07-29 — Fused MoE glue: an MoE layer goes from 24 dispatches per token to 14, and 35B decode from 92.6 to 102.8 tok/s

**Context.** With the DeltaNet layers fused, the 35B-A3B's remaining decode cost is the
MoE half, and it was launch-bound rather than bandwidth-bound: 24 dispatches per layer
at seq==1, 960 per token across the 40 layers, of which exactly 8 are real matmuls.
Everything else is glue — a softmax, an arg-sort, a gather, a sum, two clamp halves that
each upload a fresh 4-byte scalar buffer, a divide, a sigmoid, three elementwise
multiplies and three adds.

**Change.** Three fusions, each bit-identical to the candle chain it replaces, behind one
kill-switch (`XWEN_MOE_GLUE_CLASSIC=1`).

`kernel_moe_router` (new `src/ops/moe_glue.metal`) takes the router logits and returns
the selected ids and their renormalized weights in one threadgroup per token, replacing
seven dispatches. `kernel_moe_epilogue`, in the same file, computes
`Σ_k down[k]·w[k] + shexp·sigmoid(gate)` in one pass over the uncombined down
projection, replacing the weighted combine, the shared-expert gate sigmoid, its
broadcast multiply and the routed+shared add — four dispatches. And the shared expert's
`silu(g)*u` now calls the existing vendored `ops::silu_mul` instead of candle's two
elementwise passes. The routed experts' `ExpertFfn` grew a `project` method returning
the down projection BEFORE the combine, so the epilogue can own the combine; the
reference oracle declines it (it scatters as it goes) and so does the f16-tile prefill
branch (its projection carries an L2 rescale the epilogue has no term for), and both
fall back to the untouched classic composition.

**The router matmul stays candle's, deliberately.** It lowers to MLX's
`gemv_t_float32_bm1_bn2_sm8_sn4_tm4_tn4`, whose K-partition is strided across lanes
(thread t owns `{i·32 + 4t + tm}`) and whose combine is a specific shuffle-down tree —
reproducible only by lifting the kernel verbatim, and even then resting on the compiler
contracting `result += vc * inter` to an fma the same way. Worse, its accumulation order
depends on the output width, so concatenating the shared-expert gate row onto the router
weight to save a dispatch would have changed that gate's bits. Both matmuls stayed
candle's; the fusion starts at the logits. This is branch (b) of the briefing's decision
tree, and it costs one dispatch out of the ten saved.

**Bit-identity held everywhere, including the two places it looked unlikely.** candle's
`softmax_last_dim` is not a max/exp/sum triple but the online Welford form, with `fast::exp`
in the merge, `fast::divide(1, d)` hoisted once per thread, and a five-step
`simd_shuffle_down` tree over `MD{m, d}` pairs (the merge op is not a built-in simd op,
so candle takes the shuffle tree, not `simd_sum`); the kernel reproduces its
compile-time BLOCKSIZE switch over a runtime width. And candle's Metal arg-sort is
llama.cpp's bitonic network, which is deterministic but NOT stable — its comparators are
strict and never consult the index, so equal probabilities do not come out in ascending
expert order. That matters more than any rounding here: a tie flip swaps an entire
expert. The network is reproduced verbatim rather than replaced by a selection
algorithm, and `router_ties_match_candle_bitwise` pins it against candle's on four
tie families (all-equal, a twelve-way plateau with eight slots to fill, a
sixteen-value block repeated across the row, and a single winner over thirty-two tied
runners-up).

**Perf.** 35B-A3B Q4_K_M, `lowpowermode 0`, warm, batch 1, `--no-draft`, 630-token
prompt, 256 decoded tokens. Interleaved arms, median of 5, per docs/decisions.md
"Measurement discipline":

| | classic | fused | |
|---|---|---|---|
| decode | 92.6 tok/s | 102.8 tok/s | +11.0% |
| prefill 925 | 2475.1 tok/s | 2488.9 tok/s | +0.6% |
| prefill 4k | 2101.4 tok/s | 2105.8 tok/s | +0.2% |

The decode arms never overlap across the five reps (classic 91.9–93.7, fused 102.2–104.3),
and an earlier independent run on a cooler machine put the same ratio at +11.1%. Prefill
is unmoved, as it should be: above `MM_ID_MIN_SEQ` the epilogue declines and only the
router kernel and the shared expert's activation fuse, against a chunk that is
compute-bound rather than launch-bound.

**A fourth fusion was built, measured, and switched off.**
`kernel_mul_mv_id_q4_K_f32_dual` computes the gate and up expert matvecs and their
SwiGLU activation in one dispatch instead of three, and is bitwise against that chain —
it runs the same `mul_mv_q4_K_sums` body twice, factored out of the verbatim ggml
`mul_mv_q4_K_impl` so neither projection's accumulation order moves. It measured
SLOWER: 99.5 against 102.8 tok/s decode, five interleaved reps apart with no overlap. It
is not register pressure — both pipelines report `max_total_threads_per_threadgroup` of
1024. It is the grid: two independent `_v` dispatches expose twice the threadgroups, and
a bandwidth-bound gather turns that into outstanding memory requests, so folding them
into one grid that walks both weight planes serially per thread costs more than the two
saved launches buy. Kept behind an opt-in `XWEN_MOE_DUAL=1` as a measured artifact worth
re-pricing on another device, not because anything runs it.

**Parity provenance needs no schema change, and that is the point of the bitwise work.**
Every shipped kernel here reproduces its chain's rounding boundaries exactly, so there
is no path for the fused/classic split to change a dump — no new provenance field, no new
pin, no grandfather clause. Both gates confirm it, reporting numbers identical to the
pre-change run: 35B ALL PASS (strict cos=1.000000, mm cos=0.999631, decode 63/63/62 of
64 with 0 mismatches, ppl Δnll=0.000791), 27B ALL PASS (strict and mm cos=1.000000,
decode 64/64 on all three fixtures, ppl Δnll=0.000330). The dense 27B never enters this
code — `DenseMlp` keeps the free `swiglu` helper untouched — and was re-gated only
because the ledger says a shared file was edited.

`XWEN_MOE_GLUE_CLASSIC=1` joined `parity-gate.ts`'s `referenceEnv()` on hygiene, so the
oracle stays a pure candle composition, and it is deliberately NOT on the strict
candidate: the strict tier now grades the FUSED glue against a reference that ran the
unfused one. Regenerating the reference under the new pin (`--regen-ref`) and re-running
returns `cos=1.000000 top5=5/5` with every other tier unchanged — the full-model
confirmation that the ops-level bitwise tests generalize past synthetic shapes.

**Tests.** 679 lib tests pass (the two deliberately-red dflash `real_file` tests
unchanged), plus 63 in the parity binary. Eleven are new: eight in `ops::moe_glue`
(bitwise router across eight geometries including a padded bitonic network and a softmax
width below candle's shared-memory threshold, the tie families, offset views, a clamp
whose floor actually binds, the bitwise epilogue across three sequence lengths with gate
logits in both sigmoid saturation tails, a `#define`-vs-Rust geometry cross-check, and
the shape/dtype contracts), two in `ops::mv_id` for the dual kernel, and
`moe::fused_block_matches_candle_bitwise`, which runs a whole synthetic MoE block
through `MoeBlock::forward` and compares it against a reference built entirely from
candle ops — the wiring test, where a swapped operand or a gate that picked up a stray
sigmoid would show up and the per-kernel tests would not.

## 2026-07-28 — Fused DeltaNet kernels: a layer goes from ~65 dispatches per token to 8, and 35B prefill from 305 to 2183 tok/s

**Context.** Three of every four Qwen 3.6 layers are gated DeltaNet, and all of them
ran the P3 reference: composed candle ops, one scan step per token. Decode spent about
65 Metal dispatches on a layer containing seven matmuls. Prefill was worse in kind, not
just degree — the scan is a Rust `for t in 0..seq` loop issuing eight dispatches per
timestep per layer, so a 512-token chunk on the 35B cost roughly 123k dispatches. This
is P8's decode-side package. The chunked (chunk 64, tri-solve) scan was explicitly out
of scope and stays open as P8b.

**What landed.** `src/ops/delta.metal` + `src/ops/delta.rs`, four kernels wired into a
`forward_fused` beside the untouched reference (renamed `forward_classic`, math
verbatim — it is still the oracle):

- `kernel_delta_conv` — causal depthwise conv, silu, and the next conv window, reading
  the carried window and the fresh qkv rows as two buffers. That kills the `cat`, and
  writing the window directly kills the `zeros_like` + `slice_set` materialization.
- `kernel_delta_ba` — `beta = sigmoid(b_raw)` and `g = ssm_a * softplus(a_raw +
  dt_bias)` from ONE `[hidden, 2·v_heads]` projection, built at load time by
  concatenating `ssm_beta` and `ssm_alpha`. Two gemvs become one. It emits the LOG
  decay and lets the scan exponentiate, folding away another pass.
- `kernel_delta_scan` — the whole recurrence, all T timesteps, one dispatch.
- `kernel_delta_gnorm` — the gated output RMSNorm.

Eight dispatches per layer at any sequence length.

**The scan's shape is the whole trick.** Value-dim columns of the state are completely
independent — `sk[j]`, `d[j]`, the rank-1 update of column `j` and `o[j]` all touch only
column `j` — so the only cross-thread folds are the two key-dim contractions. A
threadgroup owns one V-head and 32 of its 128 columns; thread `(r, jl)` holds 32 state
rows of one column IN REGISTERS for the entire scan. The state is read once and written
once no matter how long the chunk is, which is what turns prefill's per-timestep
dispatch storm into a single launch. Consecutive threads in a simdgroup share `r` and
take consecutive `j`, so both state passes are contiguous 32-float runs. q and k are
read straight out of the conv output with the tiled K-head mapping (V-head `h` reads
K-head `h % k_heads`) and L2 clamp-normalized in the load stage, so the reference's
materialized tile-and-broadcast disappears too.

**Measured, warm (`XWEN_BENCH=1`), batch 1, greedy (`--temp 0 --seed 7`), `-n 128`.**
Power: `pmset -g` reports `lowpowermode 0` and exposes NO `highpowermode` key on this
machine, so low-power mode is confirmed off but the High Power tier is neither
confirmed nor available — do not read these as laguna's "full power" anchors. Protocol:
fused and classic runs INTERLEAVED, three reps per arm, median reported — see the
measurement-discipline note below, which is the reason these numbers are not the ones a
naive sequential matrix produces.

35B-A3B Q4_K_M:

| prompt | prefill classic | prefill fused | decode classic | decode fused |
|---|---|---|---|---|
| 596 tokens | 305.4 tok/s | **2183.2** (7.15x) | 57.8 tok/s | **91.2** (1.58x) |
| 1929 tokens | 300.3 tok/s | **2274.1** (7.57x) | 56.6 tok/s | **88.0** (1.55x) |

27B dense Q4_K_M, which had no perf numbers at all before today:

| prompt | prefill classic | prefill fused | decode classic | decode fused |
|---|---|---|---|---|
| 596 tokens | 77.3 tok/s | **290.4** (3.76x) | 14.3 tok/s | **19.0** (1.33x) |
| 1929 tokens | 77.9 tok/s | **209.3** (2.69x) | 14.3 tok/s | **17.9** (1.25x) |

The 27B's decode gain is smaller (1.25-1.33x vs the 35B's 1.55-1.58x) because 64 dense
SwiGLU layers at hidden 5120 dominate its per-token budget; dispatch count was never
its problem. Its per-rep spread is also visibly wider than the 35B's — the 35B's
classic arm repeated at 303.2/305.4/305.6 tok/s prefill, while the 27B's fused 596-token
decode walked 21.7/19.0/17.9 across its three reps as the machine heated.

**A measurement finding worth more than the numbers.** The first pass of this matrix ran
the eight cells sequentially and reported the 27B at 13.9 tok/s decode and 131.3 tok/s
prefill. A cooled, interleaved re-run of the identical binary put the same cells at 19.0
and 290.4. The whole matrix drifts 20-35% slower over roughly ten minutes of continuous
GPU load, uniformly across BOTH arms and both checkpoints, and `pmset -g therm` records
nothing while it happens — the only tell is the control arm moving too. So a sequential
A/B on this machine is not an A/B; the arms have to be interleaved. Recorded in
decisions.md "Measurement discipline", along with the sibling trap found the same hour:
`pgrep -f "logits-dump"` matches the argv of the shell running the check, so it reports
a model process that does not exist (this bit parity-gate's own preflight too — use
`pgrep -x`).

**This is the first vendored kernel family that is not bit-identical, and that changed
the parity gate.** Every earlier fused kernel reproduced its candle chain's rounding
boundaries exactly, so its `*_CLASSIC` pin was pure provenance discipline. The scan
cannot: the reference contracts k and q against the state with a candle gemm and
normalizes q/k with a candle reduce, and the kernel partitions both across threads —
that is the point. Reassociating an f32 sum is not something a kernel can undo. So
`XWEN_DELTA_CLASSIC=1` is now pinned on BOTH sides of the strict tier (with the fused
scan on, strict stops being a bitwise tier), and a `delta` provenance field
(parity_schema v6, grandfather `classic`) proves which path each dump ran. The cached
reference dumps were written at v5 and grandfather correctly — every reference in both
gate runs was reused, no regeneration.

The other three kernels ARE bitwise, using block-scope `fp contract(off)` /
`reassociate(off)` so the scan stays free to contract into fma — its two inner loops
are the entire prefill cost, and file-scope pragmas (the sibling glue files'
convention) would have doubled their instruction count.

**Parity: ALL PASS on both checkpoints.** 35B-A3B — strict 1.000000, mm 0.999631
(was 0.999540 pre-change: slightly *better*), decode 63/64, 63/64, 62/64 agreements
with 1, 1 and 2 excused near-ties and zero non-excused mismatches, ppl Δnll 0.000791.
27B — strict 1.000000, mm 1.000000, decode 64/64 on all three fixtures with zero
excusals, ppl Δnll 0.000330.

The 35B result reproduced **three times across two independent builds**, matching to
every digit each time — twice here, and once accidentally on the parity owner's build
(an unguarded `import.meta.main` in parity-gate.ts fired a whole four-tier run as an
import side effect, since fixed). The fused kernels are deterministic: same dumps, same
cosines, same agreement counts, same Δnll, regardless of who compiled them.

The one number that moved in the wrong direction is perplexity, and it is worth stating
precisely because it is the only place the kernel's fidelity cost is visible at all.
The 35B's ppl delta went 0.000511 → 0.000791 and the 27B's 0.000221 → 0.000330 —
+55% and +49%, proportionally the same on two different architectures. And the SIGN is
systematic: the candidate is worse (higher NLL) in all four measurements, so this is
bias, not symmetric rounding noise. Everything else about the two candidates was
identical, which makes the attribution clean — that is a real, measured cost of the
fused scan, still comfortably inside the 0.002 bound.

The floor stays at 0.002 (parity owner's call, and the right one): `max(3 × measured,
0.002)` is a one-time floor-SETTING heuristic anchored to the reference-scan baseline,
not an invariant to maintain against whatever the candidate currently measures.
Re-fitting it to 0.000791 would widen the bound to fit the change under test, and a
bound re-fitted to each new implementation ratchets outward forever and catches
nothing. So the constant deliberately no longer reproduces from `3 × measured` — it is
tighter and more sensitive than the recipe would now give. Trip-wire: from 0.000791, a
further ~2.5x rise fails the gate. Since the 35B mm cosine went the OTHER way
(0.999540 → 0.999631), perplexity — not cosine — is the number to watch on further
DeltaNet kernel work.

The 35B's long-mixed fixture also picked up two near-tie excusals it did
not have before (margins 0.0161 and 0.1391 — both would clear even the standard 0.5
window), which is exactly where you would expect them: long-mixed is the fixture that
carries the DeltaNet recurrence over 600+ tokens.

**Greedy output is not preserved at longer prompts, by construction.** At a 596-token
prompt, fused and classic produce byte-identical 128-token greedy continuations on both
models. At 1929 tokens the 35B shares 69 words and then forks at a near-tie; the 27B
stays identical at both lengths (dense, no router, far less tie-prone). This is the
expected consequence of reassociated f32 sums, not a kill-switch bug — and it is why
the decode tier grades against the llama.cpp-anchored oracle with a near-tie rule
rather than against the previous build.

**One deliberate carve-out.** A `seq > 1` chunk under an armed rollback checkpoint
stays on the reference scan: the one-dispatch scan can only report the state after the
LAST token, and an armed layer needs one per token for the trail. Single tokens still
take the fused path even when armed, since their only state IS the final one — so spec
decode's per-token verify steps keep the win and only a batched verify forward pays.

**Tests.** 660 lib pass / 2 known-red (the dflash `real_file` pair, P9's), 63 parity
pass (was 60). Eleven new: four per-kernel tests at both shipped geometries (16/48 and
16/32 at head dim 128) with the conv and beta/decay ones asserting `f32::to_bits`
equality against the reference chain, a no-mutation and a streaming test for the scan,
shape/geometry rejection, offset-view handling, a block-level fused-vs-reference test
that grades the kernels as a package, and three parity rejection tests for the new
`delta` pin. The scan test was mutation-checked: flipping the K-head mapping from tiled
to interleaved moves its relative L2 from ~1e-6 to 1.37.

**Hardening follow-up, 2026-07-29.** Guards, an assertion mechanism, and doc
corrections on top of the kernels above. Nothing here moves a computed value; the
kernel math, barriers and indexing were re-read and stand.

- *Geometry, asserted twice over.* The scan's threadgroup shape lived as `#define`s in
  delta.metal (which index the state slice) and as independent Rust constants in
  dispatch.rs (which size the grid), with nothing tying the two together — drift meant
  silent out-of-bounds device writes. delta.metal now carries `static_assert`s for the
  three relations, and `scan_geometry_matches_metal` parses the `#define`s out of the
  source and compares them against dispatch.rs's copies.
- *Simd width.* Both the scan (`red[2][DELTA_D/32]`, indexed by simdgroup index) and the
  gated norm assume 32-wide simdgroups. `check_delta_simd_width` reads
  `threadExecutionWidth` at pipeline setup, so a device that ever differed fails at load
  instead of quietly folding the wrong lanes.
- *Empty chunks.* `delta_ba` and `delta_gnorm` accepted `seq == 0` and encoded a
  zero-dimension grid; they now bail like `delta_conv` and `delta_scan` already did.
- *Provenance is observed, not assumed.* `LinearAttnBlock::forward` also falls back to
  the reference scan on a non-128 head dim, a non-Metal device, or an armed multi-token
  chunk — none of which the environment shows — so an env-derived `delta` field could
  stamp "fused" on a dump that never dispatched a delta kernel, and the bounded tiers
  would grade the reference against itself and pass on nothing. Two `AtomicU64`s now
  count layer forwards down each path (`linear_attn::delta_path_counts`), and
  logits-dump derives the field from them, refusing to write a dump whose observed path
  contradicts the environment or splits across both. The field's value vocabulary is
  unchanged, so parity_schema and the gate's checks are untouched.
- *Docs that overclaimed.* TWO of the four kernels are bitwise, not three: the gated
  norm reassociates its sum of squares through `simd_sum` and grades at 2e-6, in the
  same class as the scan. The `math_mode(fast)` pragma pins the math mode but NOT the
  fast-vs-precise math-function compile option (this source compiles with no
  `MTLCompileOptions`), so what holds the intrinsics to candle's rounding is the pair of
  on-device bitwise tests, not the pragma. `ba_matches_reference_bitwise`'s decay
  assertion exponentiates through candle on both sides, so it grades `g` — not the
  scan's fast-math `exp`. And docs/parity.md's manual env table omitted
  `XWEN_DELTA_CLASSIC=1` from the reference and strict-candidate rows, so the documented
  by-hand procedure would have failed the tier the script passes.

One new lib test (`scan_geometry_matches_metal`), taking the delta family to nine, plus
four empty-chunk rejections folded into `shape_and_geometry_errors`; 63 parity
unchanged, since the `delta` field's value vocabulary did not move. The `static_assert`s
were mutation-checked out of band: halving `DELTA_TG_COLS` fails two of the three at
compile time.

## 2026-07-28 — Sampler tail: 0.82 → 0.41 ms/token, by moving the softmax off the CPU

**Context.** The per-token sampling tail was suspected of costing multiple milliseconds
of the ~16.9 ms decode budget. The bench that would have shown it (`sampler_decode_bench`
in moe.rs) was still carrying laguna's shapes — hidden 3072, vocab 100352, top-k 10,
47 MoE layers — so it had never measured Qwen geometry. Fixed first: the whole
`decode_bench` constant block now reads 35B-A3B (hidden 2048, expert_ff 512, top-k 8,
vocab 248320, 40 MoE layers), and the tiled `[VOCAB, HIDDEN]` synthetic tables tile at
512 rows because 1024 does not divide 248320.

**Measured, then fixed.** Baseline at real width: 0.819 ms/token for the whole draw, of
which 0.204 ms is the GPU→CPU copy of the 993 KB logit row and 0.615 ms is CPU work.
The CPU work was candle `LogitsProcessor::sample_topk_topp`: a full-vocabulary
temperature divide, a full softmax (three more passes plus a 993 KB allocation), a
`to_vec1`, and a `select_nth_unstable_by` over 248320 indices with an indirect
comparator. Micro-benched in isolation, the exp pass alone is 0.347 ms and the
`select_nth` 0.270 ms — together they are the tail.

**Change.** sampler.rs no longer uses `LogitsProcessor`; it owns its RNG and its
filtering. Two things moved. The full-vocabulary softmax now runs on whatever device
holds the logits — one Metal kernel instead of 248320 CPU `expf` calls — and the draw
reads back probabilities instead of logits, so the number of bus crossings is
unchanged at one. The candidate set comes from a streaming top-k (one comparison per
entry against the running k-th best, insertion on the order of `k·ln(n/k)` times)
instead of `select_nth` over an index vector. After: 0.406 ms/token, CPU work 0.206 ms.
0.41 ms/token back, ~2.4% of the budget, ~1.5 tok/s at the 59 tok/s baseline.

**The op-order finding.** candle's `TopKThenTopP` is temperature → full softmax → top-k
→ top-p, and its top-p cut is measured against FULL-vocabulary probabilities: it
compares the running cumulative mass of the top-k survivors to `top_p` without
renormalizing them first, and skips the cut entirely when the top-k mass is already at
or below `top_p`. llama.cpp and HF transformers both do the opposite — truncate to k,
renormalize over the survivors, then cut. The two disagree whenever the top-k set does
not hold nearly all the mass. This rewrite deliberately preserves candle's order, so no
distribution changed; it is why the fast path still needs a full-vocabulary softmax
rather than a 20-wide one. The divergence from llama.cpp is a ledger item, not a fix
made in passing.

**What did change: seeded token streams.** candle's candidate list came out of
`select_nth_unstable_by` in unspecified order; this one is sorted by descending
probability. A weighted draw maps its single uniform through the cumulative weights, so
the same uniform now lands on a different token. The distribution is identical, the
seeded sequence is not. Nothing depended on it — the parity gate is greedy end to end,
and the argmax path is untouched (still a CPU first-maximal scan). Pinned by two new
tests: the candidate set is compared id-for-id and weight-for-weight against a literal
transcription of candle's filtering, and 40000 draws are compared against the real
`LogitsProcessor` as a live oracle.

**Verdict.** The bench measures Qwen now, which was the precondition for any of this.
The remaining 0.41 ms is 0.20 ms of readback (mostly command-buffer sync, not copy) and
0.11 ms of streaming top-k; taking the selection to the GPU so only k values cross the
bus is the next lever, and a ledger item.

**Review follow-up, 2026-07-29.** Two reviewers went over the replacement. Nothing they
found moves the distribution on a well-formed row; three of the four are about rows that
are not well-formed, and the fourth is a claim the tests were making too broadly.

- *The padded tail was drawable.* The sampler read the whole logit width, but the output
  layer is wider than the tokenizer (248320 rows against 248070 encodable ids) and the
  rows between them decode to nothing. A padded row winning a slot puts a textless id
  into the stream. `Sampler::new` now takes the encodable bound — `tok.vocab_size()` at
  the one production construction site, never a literal — validates it against the width
  of every row it is handed, and narrows the row to it before anything else runs. The
  narrow is a view, so this is cheaper than what it replaced rather than an extra pass,
  and it keeps the padding out of the softmax denominator as well as out of the
  selection, which is what leaves the two softmax backends (device on the fast path, CPU
  on the controlled one) looking at identical values.
- *NaN was being skipped.* NaN loses every ordered comparison, so the argmax scan walked
  past a corrupt row and returned the best of the survivors — a silent answer where
  `LogitsProcessor` had at least pinned index 0. Every path now errors on a NaN, greedy
  included, which is the one the parity gate runs. `-inf` is not corruption (it is how
  the controls exclude an id) and stays skippable. The streaming top-k catches its NaNs
  inside the branch it already takes for a genuine improvement, so the hot loop still
  costs one comparison per entry.
- *The tie-break contract was overstated.* `top_k_desc`'s strict `>` against the floor
  gives lowest-id-wins at an exact top-k boundary tie; candle's `select_nth_unstable_by`
  leaves the same case unspecified. The determinism is the better contract — it is what
  makes the candidate set a function of the probabilities rather than of the traversal —
  so it is now claimed as such: distribution equivalence for untied inputs, deterministic
  low-id selection at exact boundary ties, and a test pinning the tie behavior itself
  without reference to what candle does with it.
- *Equivalence was mostly tested against a transcription.* The broad oracle is a
  hand-copy of candle's filtering, so a shared misreading would pass; only one narrow
  case ran against the real `LogitsProcessor`. That comparison is now a matrix — widths
  64 / 2048 / the checkpoint's own 248320, k of 1 / 20 / 64, top_p of 0.5 / 0.95 / 1.0,
  over flat, peaked and exactly-tied rows, 81 cases, ~4 s in release. Untied rows assert
  containment (candle must never draw an id this sampler excluded) plus matching
  frequencies; tied rows assert only eligibility, since candle's answer there is
  unspecified. Mutation-checked: shifting the top-p threshold by 30% and flipping the
  tie-break direction each fail it.

Five new lib tests, 666 passing. The one behavior change a user could observe is the
padded tail no longer being drawable, which was never a legitimate outcome.

## 2026-07-28 (later still) — Second-model review: one finding, and it was our own docstring overclaiming

**Context.** The external-model reviewer (GPT-5.6 via codex, ~35 min, xhigh) ran over
the full retarget alongside the two Claude reviewers. It cleared every model-math trap
independently (interleaved gate split, partial rope, rollback indexing, router order,
chunk carry) and missed the tool-call parser bug the integration reviewer caught — the
two reviewer families found disjoint issues, which is the argument for running both.

**Change.** Its one finding (MEDIUM): `kv_rollback`'s docstring promised caches
"byte-identical to ones that only ever advanced over the committed tokens" — but the
q8/f16 dual-storage split makes cache-mutating projections partition-dependent
(a verify batch of 9+ tokens runs the f16 plane where one-token decode runs q8 GEMV),
so the counterfactual-identity half of that promise is false in exactly the
speculative case it describes; the parity gate cannot see it because it runs one fixed
partitioning. Resolution: the mechanism is kept (the bytes-halving is why dual storage
exists), the CONTRACT was corrected — restore is a bit-exact replay of recorded bytes,
cross-partition agreement is numeric and parity-gated, `XWEN_ATTN_DEQUANT` pins one
representation when partition-independence matters. decisions.md records it under
Kernel policy; drift magnitude at the 8↔9 boundary is a ledger item, unmeasured.

**Verdict.** No code behavior changed; a claimed identity was demoted to the observed
one, per this project's own first rule. Review pass complete: three reviewers, one
critical fix (tool-call parser), one contract correction, model math clean.

## 2026-07-28 (late night) — Serve integration fixes: the tool-call parser was reading `:` and `;` as span markers

**Context.** An adversarial review of the inherited serve/ tree against the Qwen
retarget turned up four integration defects. One is severe and had no chance of being
caught by the suite that covered it.

**The headline: `serve/engine.rs` opened tool-call spans on ordinary punctuation.** The
span parser carried laguna's token ids as literals, `TOOL_CALL_OPEN: u32 = 25` and
`TOOL_CALL_CLOSE: u32 = 26`. In Qwen's vocabulary 25 is `:` and 26 is `;`; the real
`<tool_call>` pair is 248058/248059, and `tokenizer.rs` has held those constants since
the fork. So for any request carrying tools, every colon the model wrote in prose opened
a span and every semicolon closed one. What followed a colon stopped being answer text
and started being parsed as a call — delivered as a fabricated tool call if it happened
to parse, silently discarded by the heal path if it did not, which is to say the reply
truncated at the first colon. Genuine `<tool_call>` tokens meanwhile fell through to the
`_` arm and reached the client as literal text. The interior grammar was laguna's too:
`<arg_key>`/`<arg_value>`, strings that are not in Qwen's vocabulary and that chat.rs
has never emitted — so even a correctly-framed span parsed to nothing.

**Why the suite missed it.** There were seventeen tests over this parser and they all
passed. Every one of them scripted the token stream by hand as `(TOOL_CALL_OPEN,
"<tool_call>")` — the constant paired with the text a correct constant would decode to.
That pairing is the bug, asserted as a fixture. The tests agreed with the parser about
what id 25 meant, and neither had ever been asked what the tokenizer thought. A test
that builds its input from the same wrong constant as the code cannot fail when the
constant is wrong; it can only fail when the code stops being self-consistent. The fix
is not more assertions, it is a different input source, so the tests now drive the
emitter over ids produced by the real embedded tokenizer — round-tripping conversations
that `chat.rs` rendered, and one hostile case that feeds prose full of `:`, `;` and
`<function=` text and asserts zero calls with byte-identical output. Under the old
constants that case produces a tool call named `name`.

**Change.** The ids come from `LagunaTokenizer` now; the interior parser reads what
chat.rs writes (`<function=NAME>`, `<parameter=KEY>\nVALUE\n</parameter>`,
`</function>`, one function per span, framing newlines stripped from values). Two
behaviors changed on purpose rather than by translation. `</tool_call>` is structural
wherever it lands, mid-value included — the template writes a literal one inside an
argument as content so it never encodes to the added token, and treating the token as
content is what let a malformed value swallow the rest of a reply. And a span that never
names a callable tool now degrades: raw text, markers included, to the client as answer
text with a logged warning (`ServeLog::ToolSpanDegraded`, counted separately from
`healed` in the per-request report). Never discard, never fabricate.

**Three smaller items.** Drafting was on by default, inherited from laguna, which made
every zero-flag `xwen generate` and `xwen serve` abort at startup — `xwen serve` before
the listener bound. The error turns out to be `missing GGUF key dflash.decoder_arch`
rather than the `decoder_arch == "laguna"` mismatch the review predicted: the shipped
sidecars have no such key, so the failure precedes the check adaptation was expected to
repoint, with `enc.aux_norm` and `blk.N.attn_gate` absent behind it. Default flipped to
off; asking for a drafter still fails loudly. `serve/config.rs` was sizing caches with
laguna's geometry — 48 KiB/token from 12 full layers × 8 KV heads × 128 head_dim, and a
72 MiB snapshot described as copies of 36 SWA rings, in a model with no SWA layer at
all. Real figures are 20 KiB/token (35B-A3B) and 64 KiB (27B), with snapshots of
DeltaNet recurrent state at a fixed 62.8/149.6 MiB; they are derived on `hub::Model` now
rather than carried as constants. And `scan_banned` was protecting the compile-time EOG
set while the decode loop stopped on the GGUF-derived one — harmless today, since the
two match on both checkpoints, but only by coincidence.

**Verdict.** 645 lib tests pass, up from 642. Two pre-existing failures remain, both
`dflash::tests::real_file_*`, both the unadapted-drafter story above; they cannot pass
until TODO.md P9 lands and were failing before this arc. The tool-call parser is the
first inherited subsystem found to be not merely unadapted but actively wrong on Qwen
input, and the way it survived — tests built from the same constant as the code — is
worth carrying into the review of every other inherited dialect layer.

## 2026-07-28 (late night) — Parity harness live (P7): both checkpoints match upstream llama.cpp, floors an order of magnitude tighter than laguna's

**Context.** Everything before this rested on cited source lines, hand-computed unit
tests, and one greedy eyeball. The engine had never been compared against another
implementation on the same weights, and the 27B had never been run at all. P7 was the
item standing between "it produces fluent text" and "it computes the right thing".

**Change.** Upstream `ggml-org/llama.cpp` shallow-cloned into `reference/llama.cpp`
and PINNED at `e9fa0781f1c25fc4fe8c86be1edc6970661ad6f0` (recorded in docs/parity.md).
This session materialized it as a plain clone and gitignored the path, reasoning that
a submodule puts the reference tree in the index and the index is the human's review
surface; a `.gitmodules` entry declaring the same path as a submodule was staged
concurrently from elsewhere, and the owner settled it that night in favour of the
submodule — the gitlink makes the oracle sha reviewable in the diff, so moving the pin
becomes a staged change someone approves, which is the property the pin exists to
have. `scripts/build-llamacpp.sh` retargeted at the path.
`tests/fixtures/parity-prompts.json` regenerated with Qwen ids from the oracle's own
`llama-tokenize --no-bos`; the SWA-specific `long-swa` fixture replaced by
`long-mixed`, 612 tokens of prose that stresses the DeltaNet recurrence instead of a
sliding window that does not exist here. `hf.ts` repointed at the two ggml-org repos
with a size selector; `parity-gate.ts` gained `--model-size 27b|35b` and namespaces
every artifact by checkpoint basename (two official files, two architectures, two
sets of floors — they must never share a parity dir or a frozen ppl fixture).
`ref-dump.sh` retargeted. `scripts/parity.ts` learned the tap-name mapping
(`refTapNames`) rather than renaming engine taps. Laguna's `reference-ppl.json`
deleted; the committed-fixture test now validates every per-checkpoint fixture.

**Two latent parser bugs in the inherited `parity.ts`, both silent corrupters.**
First, the node-header regex captured names as `(\S+)`, so headers with spaces
(`cache_r_l0 (reshaped)`, `(view)`) were skipped and their value rows were attributed
to the previous node — which kept that node's `sum` but replaced its sampled row with
an unrelated tensor's. Symptom: `attn_norm-0` reporting `rowRelL2 = 2.29e+6` while its
values were in fact digit-identical to the oracle's. Second, `FLOAT_RE.test(line)` on a
shared `/g` regex advanced `lastIndex`, dropping every other value row. Neither would
ever produce a false PASS, but both produced convincing false divergences — the first
one cost a real detour before it was pinned down.

**Result.** Both checkpoints agree with upstream on identical GGUF weights. The final
four-tier run with the calibrated constants in place is `ALL PASS (6 graded)` on each —
42 s warm for the 35B, 2.0 min for the 27B, since the Reference dumps are cached and
only the candidates regenerate.

Track A (first-divergence bisection, 35B, code-short, 242 taps compared): no cliff.
The sampled-row rel-L2 profile is smooth and *flattens* rather than compounding —
`l_out` runs 1.8e-3 at layer 0, 1.2e-3 at 7, 2.4e-2 at 23, 1.4e-2 at 39 — and the
final-logits sampled cosine is 0.999995. Individual `sumRelErr` spikes up to 1.9e-1
are near-cancelling residual sums, not divergences; their own `rowRelL2` stays in the
neighbourhood's band.

Track B, `bun scripts/parity-gate.ts`, 35B-A3B Q4_K_M:

| tier | fixture | result |
|---|---|---|
| strict | code-short | PASS, cosine 0.999999861, top-1 = ref, top5 5/5 |
| mm | code-short | PASS, cosine 0.999539782, top-1 = ref, top5 5/5 |
| decode | code-short | PASS, 63/64 agree, 1 excused near-tie (0.0040 logit) |
| decode | text-mixed | PASS, 62/64 agree, 2 excused (0.5567, 0.2606) |
| decode | long-mixed | PASS, **64/64 agree**, 0 excused |
| ppl | — | PASS, Δmean_nll 0.000511 (fused 1.694170 vs reference 1.693659 over 4218 tokens) |

**The 27B's first forward was correct.** No bisection was needed: it loaded (18.2 GB
resident), prefilled 58 tokens in 9.6 s cold, and produced a top-5 that tracks the
35B's on the same prompt. Its parity numbers are *better* than the 35B's across the
board — strict bitwise 1.000000000, mm ≥ 0.999993294, decode **64/64 with zero excused
near-ties on all three fixtures**, ppl Δ 0.000221 (fused 1.748093 vs reference
1.747872) — with the caveat that on a dense model the
strict tier is near-vacuous, since `--moe-impl reference` and `fused` run the same
`DenseMlp` and the strict env pins everything else classic on both sides. The 27B's
real signal is the mm/decode tiers, which exercise the f16 attention path and the
fused glue.

**Floors, calibrated across both checkpoints and all three fixtures** (the constants
are global, so they are set under the WORST observed value): `COS_MIN_STRICT = 0.9998`
(worst achieved 0.999894, 35B long-mixed) and `COS_MIN_MM = 0.999` (worst achieved
0.999540, 35B code-short — ~1.7x the observed prompt-to-prompt spread). Both are an
order of magnitude tighter than laguna's 0.9955 / 0.985: these kernels track the
oracle much more closely on this architecture, largely because the Qwen Q4_K_M mix
keeps attention, ssm and shared-expert weights at q8_0.

**Verdict.** The engine is parity-validated. This is the checkpoint that makes every
later kernel change checkable instead of hopeful, and it is the gate P8's DeltaNet
kernels will be graded against. Four things are explicitly NOT proven and are in the
ledger: Track A cannot localize inside a layer (the tap set is still laguna's six —
no DeltaNet core, router logits, or shared-expert gate taps, which would need plumbing
in the model-math files); `provenance.flash` says `"fused"` while `flash.metal` is
compiled at head dim 128 and cannot serve Qwen's 256, so prefill is really candle sdpa
with a materialized mask; the dense strict tier as noted above; and the `_Q8` widened
bands were inherited, not recalibrated — measured worst per-step l2 deviation is
1.0211 against a 1.5 band (far too loose), while the near-tie window genuinely needed
its widened 1.0 (text-mixed step 15 excused at 0.5567).

## 2026-07-28 (night) — First real-weights decode: correct output, clean stop, 59 tok/s on the 35B-A3B

**Context.** Everything to this point rested on cited source lines and hand-computed
tests; nothing had touched real weights. The 20.4 GB ggml-org Q4_K_M finished
downloading and the release binary was fresh.

**Change.** None — this is the first execution of the P2-P6 stack:
`xwen generate -p "What is 2+2? Answer with just the number." --temp 0 --no-draft`.

**Result.** Correct end to end on the first attempt: a coherent five-step thinking
block, `</think>`, the answer `4`, and a natural stop on `<|im_end|>` (143 tokens
emitted with 369 of budget left — the stop list works). Load 2.8-3.0s via the mmap
path, 19.2 GB resident as predicted (weights 19.0 + KV 0.2 + recurrent state 0.1 at
max_ctx 8192). Warm numbers, power mode unmeasured: prefill 23 tokens at 167.6 tok/s,
decode 58.6-59.0 tok/s. The 8.76s cold-run prefill was Metal pipeline compilation, not
the DeltaNet scan — the "will look like a hang" warning in the P2-P4 entry overstated
short-prompt cost; long-prompt prefill through the per-token scan remains unmeasured
and P8 still owns it. Coherence at temp 0 is strong indirect evidence for the whole
trap set (tiled V-heads, norm baking, gate split, router order): any of those wrong
produces fluent garbage, not correct arithmetic under a formatting constraint.

**Verdict.** The engine runs Qwen 3.6 correctly by the greedy-eyeball half of the
fallback gate. Not yet parity-validated — P7 is unchanged as the next honest
checkpoint. Decode at ~59 tok/s pre-kernel-work is 2.7x laguna's decode on 2.7x
fewer active params, i.e. exactly the bandwidth story, and the DeltaNet reference's
~240 extra dispatches per token are not yet visibly the bottleneck at this length.

## 2026-07-28 (evening) — ChatML port, tokenizer swap, and a latent constrain-width bug (P5+P6): suite fully green, 642/0

**Context.** The fork still rendered Laguna's angle-bracket template over Qwen's vocab,
tokenizer.rs carried Laguna ids, and two tests failed on vocab assumptions. hub/CLI
still resolved poolside checkpoints.

**Change.** chat.rs rewritten against the vendored chat_template.jinja (ChatML, tools
preamble, XML-ish call format with string-args-raw, collapsed tool-result turns, the
last_query_index thinking-retention rule, open-`<think>` generation seeding), keeping
laguna's content/structure separation; typed `ChatError` mirroring the template's raise
cases plus one deliberate divergence (tool-result-first refused — the template emits a
malformed boundary there). tokenizer.rs repointed at Qwen ids and made the single owner
of every token id, exposing both vocab sizes (248070 encodable / 248320 logit width).
hub.rs got a per-model checkpoint table (`--model-size 27b|35b`, 35b default,
API-verified filenames); sampling defaults aligned to 1.0/20/0.95. The constrain trie
is now sized to the logit width — which surfaced and fixed a latent bug where every
constrained serve request would have died on a 248,096-bit mask against 248,320 logits.
Control-token safety under grammars turned out to rest on toktrie's angle-bracket
heuristic rather than the special flag; documented and pinned by a full-range sweep
test. Design calls recorded in decisions.md "Tokenization, chat, tool calls".

**Result.** 662 lib tests: 642 pass, 0 fail, 20 ignored (perf benches). The five
template vectors are byte-exact; a differential harness ran 20,000 fuzzed conversations
plus exhaustive role-shape and Unicode-whitespace sweeps against the reference jinja
with zero divergences. Adversarial review by a second model family found two real bugs
pre-merge (silent second-system drop; trim whitespace set), both fixed.

**Verdict.** The prompt/token surface is done and independently verified. What remains
before trusting the whole stack is a forward pass against real weights — the 35B
Q4_K_M was downloading as this entry was written.

## 2026-07-28 — Model core retargeted to Qwen 3.6 (P2-P4): config, loader, DeltaNet reference, attention and MoE

**Context.** The mechanical fork built green but still computed Laguna: 48 uniform
attention layers, a sliding-window ring every fourth layer, a softplus per-head output
gate, and a sigmoid/bias/scale MoE router. P2-P4 replaced that core wholesale with the
Qwen 3.6 hybrid, on the critical path for everything downstream. Two research agents ran
alongside the implementation — one extracting llama.cpp master's `qwen35.cpp` /
`qwen35moe.cpp` / `delta-net-base.cpp` graphs, one range-parsing the shipped ggml-org
GGUF headers — so every load-bearing form was written first from the CLAUDE.md cheat
sheet and then confirmed or corrected against a cited source line.

**Change.** `LagunaConfig`/`LagunaModel` → `XwenConfig`/`XwenModel`. config.rs parses
both archs, rejects anything else, and derives per-layer `LayerKind::{Full, Linear}`
from the `full_attention_interval` key rather than a hardcoded 4. New `linear_attn.rs`
holds the gated-DeltaNet layer as a frozen oracle in the `ReferenceExperts` sense:
composed candle ops, recurrent form only, fp32 state, one sequential scan step per
token. attention.rs gained the double-width `attn_q` with its per-head interleaved gate
(strided split, not a halving of the row), QK-RMSNorm over 256 dims, partial NEoX rope
at n_rot 64 / theta 1e7, sdpa scale 1/√256, and an ELEMENTWISE `sigmoid(gate)` — 4096
independent values per token, not one scalar per head. moe.rs swapped the router for
softmax-over-all-256 → top-8 → renormalize with the f16 floor, no selection bias and no
weight scale, and gave the shared expert its scalar sigmoid gate. kv_cache.rs grew a
third `LayerCache::Linear` variant carrying conv window and delta state in f32 through
checkpoint, rollback, snapshot and the on-disk framing; the SWA ring machinery is left
in place but nothing on the model path constructs it. model.rs dispatches per layer and
loads the pre-MLP norm from `post_attention_norm`. The mmap/no-copy loader, ExpertStack,
QLinear and the dual-storage attention planes were not touched — only the name table.

**Result.** `cargo build` and `cargo test --no-run` green. 659 lib tests: 637 pass, 20
ignored, 2 fail — `generate::ban_scan_catches_every_em_dash_token` and
`constrain::walks_a_valid_document_and_completes`, both in tokenizer/constrain code this
arc never touched, and the second is the already-logged `<think>`-is-not-special trap.
23 tests are new, including the three the recurrence needed: the delta rule walked
against a scalar f64 reimplementation of the update equations at head dim 2 over three
tokens, the gated-norm ordering pinned with a non-uniform weight and opposite-sign gate
factors, and conv-state continuity checked by feeding seven tokens one at a time versus
one batch. Six corrections came out of the ground-truth pass: no fused `ssm_ba` (two
separate tensors), `ssm_conv1d.weight` is 2-D not 3-D, `full_attention_interval` exists
and should be read, `value_length` exists and should be asserted square, neither
`eot_token_id` nor `eom_token_id` exists so the second stop id is a named constant, and
the L2 norm is ggml's clamp form rather than HF's rsqrt form. The tiled-versus-
interleaved k-head broadcast — the one assumption that could have silently corrupted
every DeltaNet layer — came back tiled, as the cheat sheet said, traced to
`ggml_compute_forward_repeat_f32`'s destination index.

One finding is a real regression rather than a correction: **the vendored flash kernel
is unreachable on this architecture.** `flash.metal` is compiled at `BD == 128` and
Qwen 3.6 is head dim 256, so prefill falls back to candle's sdpa with a materialized
`[1, n_head, seq, k_seq]` f16 mask — precisely the allocation laguna's flash path was
written to avoid. Correct but slower, and now a ledger item alongside the deleted
attention benches and the DeltaNet rollback trail's ~1 GB verify-walk cost.

**Verdict.** The engine computes Qwen 3.6. Nothing has been run against real weights
yet, and it should not be mistaken for validated: the fallback gate (reference unit
tests plus a greedy eyeball) has only had its first half satisfied, and prefill through
a per-token sequential scan will make that first real load slow enough to be
mistaken for a hang. The next honest checkpoint is P7's parity harness; until it lands,
every claim here rests on cited source lines and hand-computed tests, not on a number
measured against llama.cpp.

## 2026-07-28 — Fork bootstrap: laguna mapped, Qwen 3.6 architecture pinned down, mechanical fork started

**Context.** xwen is a manual fork of ../laguna (crate `maxuna`, ~72k lines: candle+Metal
GGUF inference engine for poolside Laguna S 2.1) retargeted at Qwen3.6-27B and
Qwen3.6-35B-A3B. Bootstrap ran as five parallel agent workstreams: laguna codebase map,
Qwen 3.6 architecture research, GGUF header survey (range-request parsing of ggml-org
files, no downloads), llama.cpp reference extraction, and the cp-based mechanical fork.

**Findings that set the design.** Qwen 3.6 is not Qwen3 — it is the Qwen3-Next-derived
hybrid: 3 gated-DeltaNet linear-attention layers per full-attention layer (full at
indices 3,7,11,…), sigmoid-gated attention output fused into a double-width q_proj
(per-head interleaved [q_h, gate_h]), QK-RMSNorm over head_dim 256, partial RoPE (n_rot
64 of 256, theta 1e7; IMROPE in llama.cpp, but provably identical to NEoX rope for
text-only), MoE with 256 experts / top-8 / softmax-then-renorm plus a sigmoid-gated
shared expert (35B), MTP head shipped as sidecar GGUFs. candle has zero support for any
of this — the DeltaNet recurrence is new code. Full config/tensor tables live in
CLAUDE.md's cheat sheet; conversion traps (norm +1 baking, tiled V-heads, ssm_a =
-exp(A_log), no ffn_norm, no ssm_in) are recorded in decisions.md "Weights and loading".

**The dflash reversal.** The plan dropped dflash.rs as Laguna-specific (a diffusion
drafter bound to a poolside checkpoint). The GGUF survey then found ggml-org publishes
DFlash sidecar drafters for both Qwen 3.6 models under the same `dflash` architecture —
the subsystem is portable after all. The removal was cancelled mid-flight; the fork
keeps dflash.rs and all drafter wiring, with Qwen adaptation tracked as its own TODO
item. Lesson, same one laguna's decisions.md preaches: check the artifact before
deleting the code that consumes it.

**Fork state.** Mechanical fork (cp-based copy, maxuna→xwen rename, MAXUNA_*→XWEN_* env
prefix, Qwen tokenizer/template vendored into reference/) running as this entry is
written; build gate is `cargo build` on the unmodified-logic tree. Docs
(this file, decisions.md, parity.md, CLAUDE.md, TODO.md) written fresh, mirroring
laguna's documentation system.

**Verdict.** Research phase complete with high confidence on every load-bearing fact
(all numbers read from shipped GGUF headers, llama.cpp master source, and live HF
repos, not from memory or blogs). Implementation fan-out next: config/loader, DeltaNet
reference, attention/MoE adaptation, chat.rs, parity harness — in that order.
