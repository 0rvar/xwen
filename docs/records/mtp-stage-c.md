# 2026-08-15 (latest, same arc) — MTP Stage C: the graph is confirmed against llama.cpp by BYTE-IDENTICAL output, the sweep moves the 3.8's defaults to p_min 0.7 / depth 4 (+44-45% code, +37-38% chat over plain), and two stage-B claims do not survive being re-run

Moved verbatim from [docs/log.md](../log.md) on 2026-09-06; the log keeps this entry's
opening paragraph and links here.


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
