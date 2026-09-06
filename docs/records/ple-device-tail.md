# 2026-09-05 — PLE gate and conv move to device for multi-token prefill: Flash-Next prefill +12-13% at 880 and 3851 tokens, on by default (`XWEN_PLE_TAIL_CLASSIC` restores the host tail)

Moved verbatim from [docs/log.md](../log.md) on 2026-09-06; the log keeps this entry's
opening paragraph and links here.


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
