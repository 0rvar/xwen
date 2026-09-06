# 2026-09-06 — Fused MoE shared expert: 5 launches per layer become 1, 35B decode +1.6%, Flash-Next +0.6%, and the probe says why: the shared expert was bytes, not launches

Moved verbatim from [docs/log.md](../log.md) on 2026-09-06; the log keeps this entry's
opening paragraph and links here.


The other half of the morning's decode item 2. The probe had just priced the shared
expert's five-launch chain at a 0.43 ms floor and the ledger had it as the next −192
launches; this entry ships the fusion, measures it on both MoE checkpoints, and finds
that the launch budget over-predicted it by a factor of five. The number is small and
positive on both checkpoints. The reading behind it is the part that changes how the
next candidate gets ranked.

**What changed.** Commit b7cd358, plus 0ed20ea, which adds a one-time host_line
"moe: shared expert fused|classic at N token(s)" so a bench can tell a fused run from a
silent fallback. Two kernels in `src/ops/moe_glue.metal`:

- `kernel_moe_shexp_gate_up` takes the Q8_0 gate and up gemvs, the SwiGLU, and the
  `ffn_gate_inp_shexp` logit as an extra threadgroup. Grid `ceil(inner/4) + 1` by `n`,
  128 threads, the activation staged in registers.
- `kernel_moe_epilogue_shexp` is the block epilogue with the shexp down gemv folded in:
  one simdgroup per output element, separate accumulators for the routed combine and the
  down dot, `dst = routed + shexp_dot * sigmoid(logit)`. `kernel_moe_epilogue` itself is
  untouched and still bitwise.

Five dispatches per MoE layer become one, so −4 per layer: −192 per token on Flash-Next
(48 layers) and −160 on the 35B-A3B (40). The fused path is taken up to
`XWEN_MOE_SHEXP_FUSED_MAX_N` rows inclusive (8); `XWEN_MOE_SHEXP_CLASSIC` restores the
five-dispatch chain and is the replay check's control arm; `XWEN_MOE_GLUE_CLASSIC` still
turns the whole epilogue path off. The design is the fused hc gate's (dd50397) applied to
a second population.

**Rounding.** BOUNDED, not bitwise: both accumulations are reassociated. Measured ~1e-6
relative L2 against an f32 host oracle at both shipped geometries (2560 hidden / 640
inner / top-10 and 2048 / 512 / top-8) at n = 1, 3 and 8. Schema v11 carries a
`moe_shexp` provenance field; the strict parity tier pins classic and the bounded
mm/decode/ppl tiers grade fused.

**The prediction going in** was +3.5-4% on Flash-Next: 192 launches at the decode
budget's ~4 µs is ~0.77 ms of a 19.4 ms token. It did not hold.

**Protocol, every measurement below.** A pinned binary built in a detached worktree under
/tmp (b7cd358 for run A, 0ed20ea for B and C), so no `cargo build` in the main tree could
swap it or its `include_str!` kernels mid-run. The 596-token `decode-630` fixture,
`generate --no-draft --raw -n 128 --stats`, three rounds with the arm order reversed each
round, 60 s idle between rounds, medians reported. `pmset -g` printed `lowpowermode 0` at
the start and at the end of the session. Nothing else was on the GPU.

### A — Flash-Next decode A/B (/tmp/ceil/shexp-ab.log)

| arm | decode tok/s (rounds) | median | prefill median |
|---|---|---|---|
| fused (default) | 50.0 51.5 51.6 | **51.5** | 1169.1 |
| `XWEN_MOE_SHEXP_CLASSIC=1` | 50.2 51.3 51.2 | **51.2** | 1169.5 |

Per round −0.4%, +0.4%, +0.8%; **+0.6% at the median**. Neutral to slightly positive, and
an order of magnitude under the prediction.

### B — 35B-A3B decode A/B (/tmp/ceil/shexp-ab-35b.log, `--model-size 35b`)

| arm | decode tok/s (rounds) | median | prefill median |
|---|---|---|---|
| fused (default) | 115.1 115.0 114.8 | **115.0** | 3009.1 |
| classic | 113.2 112.4 113.5 | **113.2** | 3009.1 |

Per round +1.7%, +2.3%, +1.1%; **+1.6% at the median**, fused ahead in every round. 115.0
is the new 35B plain-decode figure. It was 114 at 0261e17 on 2026-08-30, and today's
classic arm read 113.2, so the session level is the same and the gain is within it.

Liveness: both checkpoints printed "moe: shared expert fused at 5 token(s)" on the pinned
binary (/tmp/ceil/shexp-live.log, shexp-live-35b.log), so both A/Bs measured the fused
path rather than a silent fallback. That line is exactly why 0ed20ea exists.

### C — The probe in decode mode, six arms (/tmp/ceil/shexp-probe.log)

`XWEN_DUP_DECODE=1` on the 0ed20ea binary, same fixture and protocol; ms/token is
1000/median. On the fused binary the `shexp` stage is kernel A alone, and the epilogue
variant carrying the down gemv sits under `moe_glue`.

| arm | decode tok/s (rounds) | median | ms/token | delta | prefill median |
|---|---|---|---|---|---|
| base, fused | 51.0 51.6 51.6 | **51.6** | 19.38 | | 1170.9 |
| fused + dup `shexp` | 50.7 50.8 51.0 | 50.8 | 19.69 | **+0.31 ms** | 1171.7 |
| fused + dup `moe_glue` | 51.3 51.3 51.1 | 51.3 | 19.49 | +0.11 ms | 1051.1 |
| classic | 51.0 51.3 51.2 | **51.2** | 19.53 | | 1171.3 |
| classic + dup `shexp` | 49.3 49.9 49.8 | 49.8 | 20.08 | **+0.55 ms** | 1172.7 |
| classic + dup `moe_glue` | 50.9 50.6 50.8 | 50.8 | 19.69 | +0.16 ms | 1051.6 |

The prefill column is unchanged by the fusion, as it must be, since the fused path is off
above 8 tokens; the `moe_glue` prefill delta of ~0.058 s at 596 tokens is the same figure
the morning run got, a consistency check on the instrument across two binaries.

**The reading, and this is the finding to carry forward: the shared expert's cost is its
BYTES, not its launches.** Three Q8_0 planes of 1.74 MB are 5.2 MB per layer, 250 MB per
token, which is 0.46 ms at the measured ~540 GB/s. Kernel A alone floors at 0.31 ms for
its 3.5 MB per layer, which works out to ~535 GB/s, so it is already at rate. The five
classic launches floored at 0.55 ms. The launches the fusion removed were therefore
bandwidth-bound and mostly hidden under traffic they had to move anyway, and, hanging off
the same input as the routed expert gathers with no hazard between them, partly overlapped
with those as well. Fusing them recovered only the gaps, ~0.15 ms on Flash-Next by the
probe's own base-against-classic difference, which is what A found end to end.

Contrast the hc gate, whose seven launches carried ~1 MB each on a strictly dependent
chain: there each launch's latency was exposed, and the ~4 µs budget held to within a
point (+9% measured against +7.8% predicted).

**So the decode budget gets a refinement** (decisions.md "Ceilings"). The ~4 µs average
per dispatch is an average over launches of very different byte weight. A fusion candidate
is worth (launches removed × ~4 µs) only when its launches carry less than ~4 µs of
traffic at rate, roughly 2 MB, AND sit on the dependent chain. Byte-bound or overlapped
launches yield their gaps only, which is a small positive rather than the budget figure.
Re-ranked by that rule, the remaining decode candidates are the MoE glue kernels (the
router kernel and the epilogue: tiny bytes, on the chain), the token-id readback sync, the
QSA tail and the GDN glue. The router projection is not on that list at all: it is not a
launch-count item but an occupancy item, 8 threadgroups for 5.2 MB, and its lever is the
vendored wide-grid f32 gemv now being implemented, unmeasured, log entry pending.

One number to note against the morning entry: the classic shexp floor reads 0.55 ms here
where it read 0.43 ms this morning, off the same five launches at the same geometry. The
duplicated arm is 20.08 ms/token in both sessions; it is the base that moved, 19.65 this
morning against 19.53 now. Session level, not a change in the stage.

### D — Correctness

Flash-Next replay check, `bun scripts/flashnext-replay.ts --control
XWEN_MOE_SHEXP_CLASSIC=1`, oracle reused (pin 6fe7498): **PASS**. Code-short 62/64 (2
excused), text-mixed 64/64, long-mixed 58/64 (6 excused), 0 hard
(/tmp/ceil/replay-shexp.log).

35B parity gate, `bun scripts/parity-gate.ts`: **ALL PASS**
(/tmp/ceil/parity-gate-shexp.log).

| tier | result |
|---|---|
| strict (classic pinned) | cos 1.000000, top5 5/5 |
| mm | cos 0.999618, top5 5/5 |
| decode | 63/64, 62/64, 61/64 agree (1/2/3 excused, 0 mismatch) |
| ppl | delta-nll 0.001179 |

Reviews: Codex (via `codex-review`) and an Opus reviewer on the diff. Neither found a
correctness bug. Every finding was a verification or documentation item: multi-block
partitions above hidden 4096 / inner 1024 are admitted by the launcher but never
dispatched by a test; the n > 1 classic comparison is built on a bare `QMatMul` rather
than the planed `QLinear` route production takes; the offset test leaves four bindings at
zero; the `fused_shexp` predicate in the docs was missing three launcher conditions; two
parity rows overstate `XWEN_SHEXP_QMATMUL` / `XWEN_MV_EXT_CLASSIC`; and the `moe_shexp`
provenance label comes from the env predicate rather than from observed execution, so it
records intent rather than what ran. A fix commit is in progress and has not landed.

**Decision: shipped on by default.** +1.6% on the 35B in every round, +0.6-0.8% on
Flash-Next, no loss anywhere, both checks pass. Open items, all ledgered: the
`MOE_SHEXP_ROWS_PER_TG` constant (4) and the 128-thread shape are UNSWEPT, and both are
one-constant A/Bs; the provenance label records the predicate rather than execution; and
the two-launch epilogue fold is cheap enough not to matter either way, +0.11 ms on the
glue stage against +0.16 classic, which is inside the noise of this protocol.
