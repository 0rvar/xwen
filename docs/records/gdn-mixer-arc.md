# 2026-08-30 (later still) — the GDN mixer arc: a per-step profiler names three targets, two of them turn out to be its own brackets, and folding the beta|alpha projection buys +4.6-4.8% on Flash-Next and +8.8% on the 35B-A3B

Moved verbatim from [docs/log.md](../log.md) on 2026-09-06; the log keeps this entry's
opening paragraph and links here.


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
