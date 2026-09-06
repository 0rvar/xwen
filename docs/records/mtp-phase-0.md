# 2026-08-15 — Phase 0 for MTP drafting on Qwen3.8-27B: the 3.6 DFlash head partially transfers but does not pay (0.86-1.02x vs the native head's 1.33-1.65x in the same session), and an MTP draft step costs 7-8.5% of a target forward

Moved verbatim from [docs/log.md](../log.md) on 2026-09-06; the log keeps this entry's
opening paragraph and links here.


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
