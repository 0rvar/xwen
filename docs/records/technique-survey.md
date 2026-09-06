# 2026-08-30 (technique survey) — Perf landscape and technique survey (research, no code): no public Apple Silicon runtime is a peer on Flash-Next, four techniques survive the cut, and candle turns out to already implement MLX-style concurrent encoding

Moved verbatim from [docs/log.md](../log.md) on 2026-09-06; the log keeps this entry's
opening paragraph and links here.


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
