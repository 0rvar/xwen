# Measurement discipline

One topic of [docs/decisions.md](../decisions.md), the index of decisions by topic; moved
here verbatim on 2026-09-06. Dated paragraphs, newest additions appended within their topic.


Inherited unchanged from laguna: state the power mode with every number, never report
first-forward prefill as steady-state, bench via the scripts with warmup, and one
~20–70 GB process at a time (2026-07-28). AMENDED 2026-08-30: `pmset -g`'s key set on
this machine is not stable — it used to print `lowpowermode` and no `powermode`, and on
2026-08-29/30 it printed `powermode 0` and no `lowpowermode`. Quote whichever line the
session actually produced, verbatim, and read it BEFORE the runs; an after-the-fact
reading does not establish what was in force during them. Neither key ever licenses a
high-power claim. AMENDED 2026-09-05: the two names are one key. With the owner's mode
switched to high performance, the bench shell printed `lowpowermode 2` and the owner's
terminal printed `powermode 2` in the same second (0 all day before in the bench
shell, on automatic); which name a shell prints is unexplained, the value is what moves.
And the mode did not move the numbers: bandwidth sweep, plain decode and 3851-token
prefill re-run in high performance mode landed inside the automatic-mode spread (log.md
2026-09-05 "Ceiling diagnosis"), so absolute figures measured on automatic stand, and a
run still quotes the line it saw.

**Achievable bandwidth is MEASURED, not quoted, since 2026-09-05: argue bytes-moved
against 537-565 GB/s, and price a dispatch at 2.5 µs of floor plus its own ramp.**
`ops::bandwidth::tests::bandwidth_sweep` (log.md 2026-09-05 "Ceiling diagnosis")
streams a 2 GB device buffer with a reduce-only read and a copy, amortized and
interleaved under the thermal protocol: streaming read 537-565 GB/s median (575-580
best rounds; 87-94% of the 614 nominal), copy ~517 GB/s of bytes touched, a 32 MB plane
528-537, and a fixed cost of 2.4-2.7 µs per back-to-back dispatch inside one encoder,
measured on planes whose bytes are free. Machine in the owner's "automatic" mode,
`lowpowermode 0`. Three consequences. First, the old "never measured, do not argue
from peak" rule is retired: a bytes/time argument divides by the measured range and
says so. Second, the 8.41 µs intercept fitted to the Q8_0 gemv on 2026-08-30 is that
kernel's ramp and tail, not the launch floor; a per-dispatch budget uses ~2.5 µs for
glue kernels, ~8 µs for gemv-shaped ones, and the decode budget's residual back-solves
to ~4 µs average over the mix (an attribution, bracketed by those two measurements,
not a third measurement). The floor is a dependent-chain figure that includes host
encode cadence: candle's encoder is `MTLDispatchType::Concurrent` with automatic
barriers between dependent dispatches, a decode step is mostly such a chain, and the
probe cannot say whether the 2.4-2.7 µs is GPU drain-and-fill or the CPU encoding
(1740 × 2.4 µs ≈ 4.2 ms against 3.7 ms of measured process CPU per token). Third, "is the GPU the bottleneck" is a measured question:
`/usr/bin/time -l` differenced between two run lengths gives CPU seconds per token
(decode 3.7 ms of a 21.3 ms token; prefill 0.6 ms of 0.885), so decode is not
CPU-bound in-process — prefill's 68% duty, sys-dominated, leaves a host-side question
open — and candle's command-buffer granularity (`CANDLE_METAL_COMPUTE_PER_BUFFER`
10 / 50 / 250) moves nothing. The stack profiler's inflation was re-measured the same
day at 2.2x on prefill (511 vs 1129 tok/s at 3851 tokens), so it ranks prefill stages
too and prices neither phase — the same rule as decode, now with the number.

**A/B perf comparisons must INTERLEAVE the two arms, and a sequential matrix is not a
valid A/B.** Measured 2026-07-28 while benching the fused DeltaNet kernels: a
back-to-back matrix of eight `xwen generate` runs drifts **20–35% slower** end to end,
uniformly across both arms and both checkpoints, over roughly ten minutes of continuous
GPU load. `pmset -g therm` records nothing while it happens, so there is no flag to
check — the only tell is that the control arm moves too. Two consequences, both learned
the expensive way. Run every arm of a comparison adjacently (F, C, F, C, …) and report
the median of each, so both arms sample the same thermal envelope; the ratio survives
drift even when the absolutes do not. And treat any absolute tok/s figure as a
warm-machine number unless it came off an idle machine — the first pass of that matrix
reported the 27B at 13.9 tok/s decode and a cooled, interleaved re-run put the same
build at 19.0.

**This does NOT touch the parity gate, and it is worth saying so explicitly.** Every
tier grades logits, agreement counts or mean NLL — all arithmetic, all thermally
invariant. A throttled run produces bit-identical dumps, just later. The only
thermally sensitive figures in docs/parity.md are the wall-clock runtimes ("42 s
warm"), which are scheduling guidance and not gate criteria. So the interleaving
protocol is a bench-work rule; the gate needs no equivalent (2026-07-28).

**`pgrep -f` is not a usable "is a model running?" guard — test what the process is
EXECUTING, not what its command line mentions.** The pattern string appears in the argv
of whatever runs the check, so `pgrep -f "logits-dump"` matches its own wrapper and
aborts over a model process that does not exist. This bit both
`scripts/parity-gate.ts`'s preflight and an ad-hoc bench guard within the same hour.
Note the failure mode is not "the wrapper is a shell" — a bun/python/make wrapper, or a
`git diff -- src/bin/logits-dump.rs`, matches just as well — so excluding `sh -c` is a
heuristic, not a fix. Two structural fixes, both sound: match the process NAME exactly
(`pgrep -x logits-dump`), or keep `-f` and filter on `argv[0]`, which is what
parity-gate ships (`isModelProcess`, unit-tested offline against captured lines — a
property `-x` cannot offer since it is opaque to the caller). `-x` does work —
`pgrep -x bun` matches 3 live processes where `pgrep -f bun` matches 15, which is
exactly the argv-only false-positive class being eliminated.

Three traps cost both agents time while establishing that, all worth knowing before
probing process state in an agent sandbox: background processes do not survive
(`nice(5) failed: operation not permitted`), so a positive-case test needs a process
started some other way; `ps -p <pid>` returns nothing even for pids `pgrep` can see, so
"is it alive?" comes back empty and reads as "it isn't"; and `pgrep zsh` returns 0 in
EVERY form (`-f`, `-l`, `-x`, bare), so probing `-x` against a shell looks like `-x` is
broken when the target is simply invisible. Pick a probe target you can independently
confirm is running and visible — `bun` works here. A guard whose failure mode is a
concurrent 20 GB load deserves both halves tested against real processes, permissive
and restrictive, not just its matcher unit-tested (2026-07-28).
EXTENDED 2026-08-08: **filtering on `argv[0]` is only correct if you first establish
which lines ARE records.** `pgrep -fl` prints one record per line as `<pid> <argv0>
<args…>`, but an argv that embeds newlines prints as extra lines carrying no pid — and
agent harnesses produce exactly that, wrapping commands as `zsh -c "cd <repo>\n<cmd>"`.
The continuation line `cd /Users/…/xwen` then has `xwen` as its second token, which
`execName` reads as argv0, and the guard aborted both parity gates over a model process
that did not exist. The fix is one predicate ahead of the matcher: a line is a record
only if it leads with a pid, and fragments are dropped because the record they belong to
is checked on its own first line anyway. Same lesson one level down from the original —
the matcher was right and the tokenization under it was not.

**Per-stage forward timing is done IN SITU, by device sync, and only its
length-DIFFERENTIALS may be read as measurements.** `src/stack_profile.rs`
(`XWEN_STACK_PROFILE`) decomposes a chunk's wall clock into the stages `run_stack`
actually runs, on the real weights in the real dispatch order, because a stage budget
assembled from synthetic microbenchmarks cannot see what the wall clock holds that no
stage claims — which is the entire question it was built for. Five design rules, each
answering a way the instrument could lie:

- Stages are bracketed by `Device::synchronize`, so a stage's total is completed GPU
  work rather than enqueue time. Adjacent stages SHARE the sync between them, so the
  brackets add one sync per stage and not two.
- Host-side gaps go to their own `inter_stage_host` bucket. The sync closing one stage
  opens the next, so an unbucketed gap would be silently charged to whichever kernel
  follows it, and per-token cost living in the glue would be mis-attributed to the
  stage after it.
- `unaccounted == 0` is an ENFORCED bracket-integrity invariant, not a result. Every
  interval inside a chunk's bracket belongs to some bucket by construction; a nonzero
  value means the brackets are wrong. Anything real shows up as a bucket.
- Phase (prefill vs decode) is DECLARED by the caller (`XwenModel::set_phase`), never
  inferred from token count. Inference has two failure modes that both occur in
  practice: a prompt whose length is ≡ 1 mod 512 ends prefill with a one-token chunk,
  and a speculative verify forward feeds a whole span while being decode.
- `XWEN_BENCH`'s warm-up pass is excluded by resetting the accumulators after it, so a
  dump never averages a cold chunk with warm ones.

**The reading discipline is the load-bearing part.** Per-stage syncs roughly DOUBLE the
prefill wall — absolute synced numbers are ~2x plain and mean nothing on their own. What
survives is the differential between two prompt lengths: the per-stage sync overhead is
approximately constant per token, so it cancels out of a length delta. Any claim taken
off this instrument must be a length-differential, never an absolute. It is built for
plain `--no-draft` generation (the speculative and server paths accumulate correctly but
print no dumps), and with the variable unset the cost at each instrumented site is one
`Option` check. `XWEN_STACK_PROFILE` and `XWEN_CHUNK_SYNC` are both stripped in
`parity-gate.ts`'s `baseEnv()` (2026-08-08).

**Retuning the drafting controller is a scripted protocol now, `scripts/retune-draft.ts`,
and its load-bearing rule is that NO cell is reused between stages.** The constants have
now been wrong about three successive cost curves (P9's reference scan, P9a's
fixed-cost-dominated verify, and the post-`mul_mv_ext` one), so the protocol is a script
rather than a procedure someone re-derives from the last log entry. The shape: stage 1 sweeps `draft_p_min` at the shipped margin, stage
2 sweeps `pause_margin` at stage 1's winner, both against a plain `--no-draft` baseline
arm, rep-outermost interleave, greedy `-n 128` under `XWEN_BENCH=1`, 3 reps, medians.
The qualification criterion is P9's and is preserved deliberately: an arm qualifies only
if it is ahead of plain's median on BOTH prompt kinds in EVERY rep, the winner is the
highest mean-of-medians among qualifiers, and a tie resolves toward the shipped value.
The no-reuse rule is the part that is easy to get wrong: carrying stage 1's measurement
of the shipped margin into stage 2 looks like a free saving and is the interleaving
error one level up — it grades two arms from different thermal epochs. Every stage-2 arm
is re-measured. Supporting requirements, each of which a review found missing: cell
identities must not collide under rounding, the child env is clean with the HF cache
root resolved once to an absolute path and `HF_TOKEN` never passed down or serialized,
there is a contention guard and a per-run timeout, and the raw JSON dump is written 0600
and atomically after every run so a sweep that dies mid-way still yields its data. The
script prints recommendations and never edits a default (2026-08-08).

**A warm-up pass that reads faster than the timed pass following it is a thermal
ordering artifact, not evidence.** The warm-up runs on the cooler chip. This looked like
evidence about allocator-pool state during the residual diagnosis and was not; the
profiler excludes warm-up from its dumps for exactly this reason (2026-08-08).

**`--stats` reports drafting against a plain baseline measured INSIDE the same run, and
its plain bucket deliberately excludes wasted drafter time.** Every round is bucketed as
plain (no draft block verified — paused, empty-draft, serial-thinking, or past the
drafter's context; one token) or drafted (a block was verified; the round's full time,
draft phase included; accepted plus bonus tokens), with full-accept as a subset of
drafted marking the ceiling a longer block could reach. The load-bearing choice is what
the plain bucket folds: `round_ms - draft_ms`, the same quantity the pause controller's
plain comparator uses, so a round that ran the drafter and then committed plain
contributes only its target forward. The bucket is therefore not "what those rounds
cost" — it is what plain decode costs on this run's text, which makes `plain_rounds /
plain_ms` an interleaved plain arm sampled in the same thermal envelope as the drafted
rounds it is compared against. That is the whole reason to build it this way: the
interleaving rule above says a cross-session ratio is not evidence, and the 27B's
between-session level shift has already forced that warning into a log entry. A
breakdown that carried in an external plain number would reproduce exactly the error the
rule exists to prevent. `est. net ±Y.Y%` follows from the same partition — all committed
tokens priced at the run's own plain rate, against `plain_ms + spec_ms + (draft_ms -
spec_draft_ms)`, three terms that cover decode-loop model time once each. It is gated on
`plain_rounds >= 8`, because a rate off a handful of rounds is noise, and it is an
estimate rather than a measurement: it assumes the plain rate the run did exhibit is the
rate a fully plain run would have exhibited, which a real `--no-draft` A/B still owns.
The per-round averages it replaced divided by all rounds and inverted the reading of a
mostly-paused run; `draft` now divides by the rounds that actually drafted (2026-08-09).
