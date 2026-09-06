# 2026-08-08 — the small-batch window reaches the attention and DeltaNet projections (verify span 8 −12.0%), and the first real controller sweep makes `draft_p_min` per-checkpoint: 0.5 on the 27B, +11-13% over the shipped 0.3

Moved verbatim from [docs/log.md](../log.md) on 2026-09-06; the log keeps this entry's
opening paragraph and links here.


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
