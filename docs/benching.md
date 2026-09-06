# Benching on this machine

The standing rules for measuring xwen on the M5 Max. Each one is here because it has
already cost this project or laguna a wrong number. The figures themselves live in
[perf-state.md](perf-state.md); the reasoning behind the protocol is in
[decisions/measurement-discipline.md](decisions/measurement-discipline.md).

## Protocol

- **Bench a PINNED binary.** Build a detached worktree under /tmp and pass `--bin` to the
  harness. A coding agent's `cargo build` in the main tree swaps `target/release/xwen`
  and its `include_str!` kernels under a running harness; on 2026-09-05 a session aborted
  on a half-written kernel.
- **Interleave the arms tightly**, A B A B, never all of A then all of B. Report medians.
- **~60 s idle between rounds**, and keep the runs short. The machine thermal-throttles
  under sustained load as a matter of course (owner's word, 2026-08-30), and duty cycle
  shows up directly: the same shape measured 23% slower in a 36 s run than in a 9 s one,
  with no thermal flag anywhere.
- **Anchor arm at the start and the end of every session**, with a >3% drift flag between
  them.
- **One model process, nothing else.** Never a test suite and never a second model
  process alongside a bench. Two 20 GB processes fit RAM but not comfort, and the 27B
  Q8_0 at 28.6 GB plus anything else is asking for GPU OOM.
- **On a machine shared with other agents, calibrate against the classic arm's known
  baseline before believing an absolute.** Three separate contended runs read 3x low in
  BOTH arms while the ratio between them stayed put.
- **Never report first-forward prefill as steady state.** llama.cpp's prefill
  thermal-boosts harder than xwen's, settling -17% against -5%, so a first-reps prefill
  ratio is not a steady-state ratio.
- **Never pipe model output through a pager.** `glance` exists; an EOF-spinning llama-cli
  once fed 88 GB into `less` on the laguna side. Scripted llama-cli needs
  `-st -no-cnv </dev/null`.
- **State the power mode next to every number**, per the rule below.

## The power-mode line

Report the `pmset -g` line verbatim as of the session, and **never claim high-power
mode**: it is not positively confirmable from either key.

The key names have caused repeated confusion. Early sessions emitted no `powermode` key
and recorded `lowpowermode 0`; on 2026-08-29 and 2026-08-30 two agents saw `pmset -g`
print `powermode 0` and no `lowpowermode` key. On 2026-09-05 it resolved: the two names
are one key printed differently by different shells. The bench shell said
`lowpowermode 2` while the owner's terminal said `powermode 2` in the same second, after
the owner switched to high performance; 0 is automatic.

The mode moved nothing measurable that day (decode 47.0 to 47.3, prefill 1140 to 1139,
streaming read +4-5% at most; log.md "Ceiling diagnosis"), so figures measured on
automatic stand.

## Rates and units

- **Use AMORTIZED rates**: batch dispatches per sync and hold the outputs alive. Never
  per-dispatch. A budget built from per-dispatch numbers sums to 127% of wall.
- **Argue bytes-moved against MEASURED bandwidth, never the nominal figure.** Measured
  2026-09-05 by `ops::bandwidth::tests::bandwidth_sweep` on "automatic" power mode,
  `lowpowermode 0`:

| Quantity | Measured |
| --- | --- |
| streaming read, median | 537-565 GB/s |
| streaming read, best rounds | 575-580 GB/s (87-94% of the 614 nominal) |
| copy | ~517 GB/s |
| 32 MB weight plane | 528-537 GB/s |
| fixed cost per back-to-back dispatch inside one encoder | 2.4-2.7 µs |

- The Q8_0 gemv's **8.41 µs intercept is kernel ramp, not launch floor**; a decode budget
  closes at ~4 µs average per dispatch.
- **Between two arms, compare bytes-moved against time.** The Q4_K FFN gemm reads 3.6x
  fewer weight bytes than the f16 one and takes 2.4x longer, which settles
  bandwidth-versus-kernel with no peak figure involved.

## What the instruments can and cannot do

**The profilers RANK steps; they do not PRICE them.** Both `XWEN_STACK_PROFILE`'s decode
stages and `XWEN_GDN_PROFILE`'s whole line are sync-inflated: two figures off the GDN line
read 2-3x high against amortized benches of the same work, and the prefill stage profiler
reads 2.2x high. Take every headline from an unprofiled run, and price a step with an
amortized bench or with end-to-end tok/s. `XWEN_GDN_PROFILE`'s decode line is the sharpest
case: it orders the steps within one run and does not time them, overstating each by
roughly its dispatch round trip — the scan reads 3.79-7.19 ms/token there against 1.43 ms
in an amortized bench of the same work. Never quote a decode figure from that line as a
cost.

**The duplicate-dispatch probe prices prefill stages in situ**, with no syncs
(`XWEN_DUP_STAGE`, log.md "Duplicate-dispatch probe"). At 3851 tokens on Flash-Next:

| Stage | In-situ cost of a 3.4 s prefill |
| --- | --- |
| expert gemms | 0.96-1.09 s (28-32% of wall, 73% of `ffn`) |
| MoE glue | 0.40 s |
| hc gates | 0.39 s (gemms 0.14) |
| GDN kernels | 0.23 s (scan 0.16) |
| shared expert | ~0 |
| unpriced | 38% of wall (projections, attention, QSA, PLE, lm_head) |

Price a prefill stage with the probe, never with the profiler or an isolated bench.

**At decode the probe reads differently** (`XWEN_DUP_DECODE`, 2026-09-06). A decode copy
has no buffer hazard against its original, so a delta above zero is only a FLOOR for that
stage, and a delta of about zero means the stage overlaps itself, never that it is free.
The probe therefore cannot price a latency-bound decode stage. Measured: the shared expert
floors at 0.43 ms of a 19.65 ms token, while MoE glue and the router projection both read
zero.

**Neither instrument sees occupancy.** A kernel that leaves the GPU mostly idle is
invisible to both the byte budget and the probe; see the third refinement under
[perf-state.md](perf-state.md)'s Ceilings, where a dispatch that read zero on the probe
was worth +10.3% once its threadgroup count changed.

## Memory

Anonymous RSS lies under mmap, because the weights are file-backed. Judge memory by
footprint (`footprint <pid>`), not RSS.
