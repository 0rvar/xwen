# 2026-09-06 — Router projection on a 256-threadgroup gemv: 35B decode 115.1 → 127.0 (+10%), Flash-Next 50.5 → 52.9 (+5%); the lever was occupancy, not launches

Moved verbatim from [docs/log.md](../log.md) on 2026-09-06; the log keeps this entry's
opening paragraph and links here.


The third decode lever of the day and the largest, and it is not a launch-count item at
all. The morning's two entries left the MoE router projection as the one decode candidate
the duplicate-dispatch probe could not price: its duplicate cost nothing, which at decode
means the stage overlaps itself rather than that it is free. What the zero did say was
that it runs at low occupancy. This entry replaces the kernel underneath it and measures
+10.3% on the 35B-A3B and +4.8% on Flash-Next, more than the fused hc gate and the fused
shared expert bought on Flash-Next between them.

**What the projection was running.** `route_logits` is `[n, hidden] × [hidden, n_expert]`
in f32: 2560 × 512 on Flash-Next, which is 5.24 MB per layer and 252 MB per token across
48 layers, 4% of a token's bytes; 2048 × 256 on the 35B-A3B. It went through candle's
`Tensor::matmul`, and at n = 1 candle lowers that to the mlx `gemv_t` kernel, which picks
tile (sm, sn) = (8, 4) with bn = 4 and covers the entire plane with **8 threadgroups of
128 threads** (the 35B's narrower plane lands on 8 threadgroups of 64). At 2..8 rows it
takes a 32×32-tile gemm on 16 threadgroups with 24 of its 32 M-rows idle. The explorer's
source reading of candle at rev 21cca0b closed the cheap routes first: the tile choice is
hardwired with no knob, a `QMatMul` over an F32 QTensor dequantizes and lands back on the
same gemv, and candle's own `kernel_mul_mv_f32_f32` receives a zero row stride from its
host code. The fix had to be a vendored kernel.

**The kernel.** Commit 24c4069. `kernel_mul_mv_f32_f32_v`, in the new `src/ops/f32.metal`,
is the f16 gemv of `src/ops/f16.metal` with float weight loads: grid `ceil(n_out/2) × t`,
32×4 threads, the same reduction, f32 throughout. On Flash-Next's 512 experts that is
**256 threadgroups against 8**. It is taken at 1..=8 rows (`XWEN_ROUTER_MV_MAX_N`,
inclusive; 0 is classic) unless `XWEN_ROUTER_MV_CLASSIC` is set, and candle's matmul still
runs above the ceiling, so prefill is untouched.

**Rounding, and why this pin is load-bearing.** Reassociation only, ~1e-6 rel_l2 from the
candle path at both geometries and at t = 1, 3 and 8. But top-k routing is discrete, so a
near-tie can flip a whole expert, and unlike every earlier switch of this kind the pin
matters: the router runs BEFORE the routing decision, not after it. The strict parity tier
pins classic and the reference oracle runs classic too. Schema v12 records `router_mv`.

The cost is memory. The router plane is now held twice, transposed for candle and as
loaded for the gemv: ~251 MB on Flash-Next (512 × 2560 × 4 B × 48 layers) and ~8 MB on the
35B. Open, and ledgered by the fix commit f5dbffd.

**Protocol, both A/Bs.** A pinned binary built in a detached worktree under /tmp at
24c4069, so no `cargo build` in the main tree could swap it or its `include_str!` kernels
mid-run. The 596-token `decode-630` fixture, `generate --no-draft --raw -n 128 --stats`,
three rounds with the arm order reversed each round, 60 s idle between rounds, medians
reported. `pmset -g` printed `lowpowermode 0` at the start and at the end of the session.
Nothing else was on the GPU.

### A — Flash-Next decode A/B (/tmp/ceil/router-ab.log)

| arm | decode tok/s (rounds) | median | prefill median |
|---|---|---|---|
| mv (default) | 52.9 53.4 52.7 | **52.9** | 1171.0 |
| `XWEN_ROUTER_MV_CLASSIC=1` | 50.5 49.4 51.1 | **50.5** | 1171.0 |

Per round +4.8%, +8.1%, +3.1%; **+4.8% at the median**, mv ahead in every round. In ms per
token that is 18.90 against 19.80, so **0.9 ms per token recovered**. The router plane's
bytes floor at rate is 0.45 ms per token, so the 8-threadgroup gemv had been costing about
1.3-1.4 ms, three times its own bytes.

### B — 35B-A3B decode A/B (/tmp/ceil/router-ab-35b.log, `--model-size 35b`)

| arm | decode tok/s (rounds) | median | prefill median |
|---|---|---|---|
| mv (default) | 126.5 127.0 127.0 | **127.0** | 3008.8 |
| classic | 115.2 114.9 115.1 | **115.1** | 3006.0 |

Per round +9.8%, +10.5%, +10.3%; **+10.3% at the median**, ahead in every round. 7.87 ms
per token against 8.69, so **0.8 ms per token recovered** on a 40-layer 2048 × 256 router
whose bytes are 2.1 MB per layer, 84 MB per token, about 0.16 ms at rate. The mlx gemv was
costing roughly 1.0 ms per token there, six times its bytes. The narrower the plane, the
worse the 8-threadgroup shape, which is why the smaller router gains the most.
**127.0 is the new 35B plain-decode figure**, against 115.0 with the fused shared expert
this morning and 114 at 0261e17 on 2026-08-30.

Both A/Bs sit on top of that fused shared expert (b7cd358), so the day's stacked levers
are **113.2 → 115.0 → 127.0 on the 35B** and **51.2 → 51.5 → 52.9 on Flash-Next**. The
Flash-Next levels drift between sessions, this session's classic arm reading 50.5 where
the morning's read 51.2, so the within-session ratios are the claims and a difference
across sessions is not one.

### C — Correctness

Flash-Next replay check, `bun scripts/flashnext-replay.ts --control
XWEN_ROUTER_MV_CLASSIC=1`, oracle reused (pin 6fe7498): **PASS**. Code-short 62/64
(2 excused), text-mixed 64/64, long-mixed 59/64 (5 excused), 0 hard on any fixture
(/tmp/ceil/replay-router.log).

35B parity gate, `bun scripts/parity-gate.ts`: **ALL PASS**, 6 graded
(/tmp/ceil/parity-gate-router.log). The strict row generates its candidate under the
classic mv fallback by construction, which is the load-bearing half of this arc's pin.

| tier | result |
|---|---|
| strict (classic mv fallback) | cos 1.000000, top5 5/5 |
| mm | cos 0.999618, top5 5/5 |
| decode | 63/64, 62/64, 61/64 agree (1/2/3 excused, 0 mismatch) |
| ppl | delta-nll 0.001179 |

Those six figures are identical to the morning's fused-shexp gate run. The mm and ppl
tiers are prefill and never take the gemv, and the strict tier pins it off, so identity
there is expected; the decode tier does take it (its three candidates are fresh schema-12
dumps carrying `router_mv: "mv"`), so identical agree counts there mean the ~1e-6
reassociation flipped no graded token, which is the expected outcome for a change that
small and not evidence about the gemv beyond no regression.

Reviews: Codex (via `codex-review`) and an Opus reviewer on the diff. Neither found a
correctness bug in the kernel or in the launcher. Every finding was a verification or
documentation item: the admission predicate does not ask the launcher's 16-byte
offset-alignment conditions; the kernel test compares against candle on the CPU rather
than on Metal; two reference-dump recipes in docs/parity.md would now produce a dump the
gate rejects, because the oracle pin is load-bearing; a code comment misdescribes the
doubled plane as ledgered and as a cost only the classic arm pays; and `mod f32` shadows
the primitive type name. The fix commit f5dbffd landed the same day, renaming the module to `f32_mv`, and
has not landed.

**Decision: shipped on by default.** +10.3% on the 35B and +4.8% on Flash-Next, ahead in
every round on both, no loss anywhere, both checks pass.

**The reading, and it names a third class of decode cost.** This item entered the ledger
as "fold the router projection into `kernel_moe_router`", a launch-count lever, and that
form of it was refuted by reading this morning. The probe could not price the projection,
because at decode its duplicate overlaps itself. The byte budget said 4% of a token's
bytes and therefore nothing worth 10%. What priced it was reading candle's grid: 8
threadgroups for a 5.24 MB plane. So beside bytes moved and launch gaps there is a third
class of decode cost, a kernel that leaves the GPU mostly idle, and it is invisible to
both instruments this project has. The way to find the rest of them is mechanical: list
every decode dispatch's threadgroup count against its bytes, and flag any plane over
~1 MB running under ~32 threadgroups. That audit is the next instrument (TODO.md).
