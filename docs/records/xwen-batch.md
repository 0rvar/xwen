# 2026-08-09 — `xwen batch`: one prefill for N items, and scored fields that report the model's confidence instead of a sampled token

Moved verbatim from [docs/log.md](../log.md) on 2026-09-06; the log keeps this entry's
opening paragraph and links here.


**Where it came from.** Two research passes opened the arc — one on the
classification/structured-extraction state of the art, one on serving-side prefix reuse
(BatchLLM, Marconi, SGLang's RadixAttention). They converged on the same shape from
opposite directions. The serving literature says the cheapest batch is one that groups
by shared prefix and prefills it once; the classification literature says the accurate
way to label a document along nine taxonomies is nine narrow questions, not one wide
one, because a single prompt asking for nine fields lets early fields condition later
ones and gives no per-field confidence at all. Decomposed classification is exactly the
workload whose items agree on everything but their last sentence. So the batch API and
the classification surface are one feature, not two that happen to share a subcommand.

**What shipped.** `xwen batch` reads one JSON request on stdin and writes one JSON
response on stdout; the checkpoint comes from the payload (`"model": "27b"` / `"35b"`)
because one request is one model's work. It renders every item, computes the longest
common prefix of the rendered TOKEN vectors, prefills it once, takes a
`take_cache_snapshot` in RAM (62.8 MiB on the 35B, 149.6 MiB on the 27B, and
position-independent), then replays it per item: restore, `sync_drafter_to`, prefill the
item's own tail, decode. The cache is request-scoped — no disk, no TTL, no eviction, no
tree. Per item: `messages`, an optional JSON schema, `thinking` (`false` / `true` / a
string to inject), an assistant `prefill`, `max_tokens`, `sampling`. A failing item
lands as an `error` on its own response and the batch carries on, because the entire
point is that one bad item must not cost the other N−1 their prefill.

**Three floors, and one is a correctness precondition wearing a heuristic's clothes.**
A prefix under `MIN_SHARED_PREFIX` (64) is not worth the bookkeeping and every item runs
cold. A single-item batch never snapshots. And the shared span is capped at
`min_len − 1`: if the LCP were allowed to equal the shortest item's whole length, that
item would decode from the snapshot position and write into positions the snapshot
covers — which `restore_cache_snapshot` forbids by construction (model.rs:716; the
full-attention layers restore by truncation and are only valid while the slots below
the position still hold the same keys). The cap is that precondition written as
arithmetic, not a safety margin.

**One design decision reversed its own premise.** The plan called for snapshotting at a
chunk-64 boundary, inherited from Marconi — where it is right, because a chunked prefill
kernel cannot materialize recurrent state mid-chunk. xwen has no such constraint: the
generation layer materializes full state wherever a prefill stops, which is what serve's
`PrefixCache` already depends on. Rounding down would have meant writing alignment logic
in order to throw away reuse that costs nothing to keep. The argument that recommended
the boundary — keep it simple — is the argument that killed it.

**The scored path: the engine writes the JSON, the model only ranks the values.**
Annotate an enum or boolean property with `include_score` and the item stops
free-decoding entirely. The runner teacher-forces the JSON skeleton it wrote itself and
scores every allowed option at each field's choice point by teacher-forced sequence
logprob; selection is argmax at temperature 0, posterior sampling honoring `top_k` and
`top_p` otherwise. The field reports `{"value", "score"}`, or with `"all"` the whole
`scores` table plus the `escape` mass that opened no option. The alternative — keep the
grammar and read confidence off the mask at the grammar's choice points — died on
llguidance 1.7.6 having no structural-position API: no captures from JSON-schema
grammars, and capture-on-completion semantics anyway. Assembly makes the selection exact
instead of inferred and needs no llguidance change. `include_score` PRESENCE routes the
item, so `false` is an annotation like any other and does not mean absent — a schema
cannot switch engines by toggling a boolean. New public API for it:
`Generator::last_logprobs_for`, a log-softmax over an encodable slice, in f64. Free
thinking still works on a scored item: it decodes until `</think>` and then assembles.

**The turn of the story is a finding from the outside-model review.** codex
(`gpt-5.6-sol`) pointed out that scoring an option's own tokens and nothing else makes a
strict-prefix option unbeatable: score is a sum of log-probabilities, every additional
token contributes something negative, so `score("low_priority") = score("low") +
(negative)` and `"low"` wins whatever the model believes. The fix is to score one
terminator token past the value — the closing quote for a string enum, the delimiter
after a bare literal — which forces the two sequences to diverge at a token they cannot
share and makes the option set prefix-free. Confirmed live on the adversarial pair
`["low", "low_priority"]`: the long option takes 0.9988 when the prompt asks for it. The
same pass hardened the seams: plan time now re-encodes every (segment, option) and
(option, segment) pair and refuses a schema whose tokenization is not the concatenation,
which rejects values that fuse with their delimiter and values needing JSON escapes.
Refusing beats scoring an escape sequence as if it were the label.

**The demo, and what it measures.** `scripts/classify-demo.ts` classifies one support
email along nine taxonomies — eight single-field items plus one compound, ten labels in
all — against embedded ground truth, on both checkpoints in sequence (one large model
process at a time). 317-token shared prefix, thinking off, greedy, `lowpowermode 0`
(this machine exposes no `powermode` key, so high-power mode is never positively
confirmable and is not claimed here):

```
35B-A3B   7/10 vs authored ground truth   total 1828 ms   (load 3470 ms)
27B       9/10                            total 6595 ms   (load 9859 ms)
```

The cache earns its keep on the same request: 1203 ms against 1989 ms with
`XWEN_BATCH_NO_CACHE=1` (an earlier run of the same payload; identical on every decisive
field). The misses are the more interesting half of the reading, and they are the reason
scores exist — the 35B's sentiment miss carries 0.56 on `mixed` against 0.43 on
`negative`, and its emotion field is a 0.51/0.48 coin flip. A consumer that thresholds
on score routes both to review instead of trusting them.

**Two numbers that are not stable, stated as such.** Replay and scratch decode the same
values but not the same bits: an item's short tail prefill routes the MoE through
`mv_id` where one long cold prefill takes `mm_id`, and across the demo batch the scores
differ in the third to fourth decimal, with one genuine near-tie (0.502 vs 0.493)
selecting differently between arms. That is the partition-dependence class decisions.md
already records, reached through a new door.
`XWEN_MM_ID_MIN_SEQ=1` forces both arms onto one kernel and makes them byte-identical,
which is how it was settled. And `escape` is honest only where the choice point is
quoted: for a bare literal the opener competes with pretty-printing whitespace, and a
boolean field was observed reporting 0.998 escape beside a 0.9986 answer. Kept,
documented on the field, refinement ledgered.

Decisions in decisions.md "Batch" (nine entries: the exact-LCP snapshot, single-level
request-scoped caching, the three floors, drafter-by-truncation, the greedy/no-thinking
defaults, assembly over grammar-mask detection, terminator-token scoring, the seam check,
the escape confound, and the score-stability class). Ledgered under "Deferred from the
batch + scored-classification arc": the serve `/xwen/v1/batch` endpoint, multi-level
prefix trees, streaming, per-token logprobs, a formal Track-B case for
snapshot-replay-vs-scratch, and the v1 schema limits.

**UPDATE, later the same day: the demo grew a multi-select taxonomy, and it produced the
arc's own thesis as a measurement.** Tagging (multi-select from a ten-tag set) went in
twice behind the same shared prefix — once as an array-of-enum through the grammar path,
once as the identical tag set expressed as a flat all-boolean scored schema, one
`include_score` P(true) per tag. Two forms of one question, so the comparison is
controlled by construction. Ground truth changed in one place first: `emotion_intensity`
was amended medium → high, because both checkpoints disagreed with the authored label
and the author accepted the models' reading — recorded because the grade moved with it,
and a graded row that quietly changed definition is worse than a miss.

Twelve graded rows now, same machine and same caveat as above (`lowpowermode 0`, high
power not claimable here):

```
35B-A3B    8/12   total 3416 ms
27B       10/12   total 10591 ms
```

**The result worth the addendum: on BOTH checkpoints the scored boolean vector beat the
free-decoded array on the identical tag set, 5/6 tags against 4/6.** Per-tag scoring
recovered `refund_request`, which joint array emission dropped on both models — the
failure the decomposed-classification research predicted (a model emitting a list
conditions each element on the ones before it and stops early), reproduced inside the
demo the arc shipped to argue for decomposition. The one remaining miss is NOT a
decomposition failure and should not be read as one: `replacement` scores P(true) 0.15
on the 35B and 0.26 on the 27B, and the tag rules never disambiguated the word — ground
truth means "a replacement occurred", both models read "a replacement was requested",
and the email's author is explicitly asking for a refund INSTEAD of another replacement.
Low confidence on a genuinely ambiguous label is the scored path behaving correctly.
Amending the tag rules would move the number; it is left alone deliberately, because the
row is the clearest live example of a score that should be low.

**And then `--compare-thinking`, which turns the demo into a controlled experiment on
reasoning itself.** One 22-item batch per model: every taxonomy twice, once with thinking
off and once with a force-injected closed `<think>` scaffold. The scaffold restates the
rubric and the evidence discipline and never states a document fact — it is a reasoning
protocol, not a hint, which is the only way the comparison stays about thinking rather
than about a leaked answer. Both arms sit behind the same shared prefix, so 22 items are
still one prefill and one snapshot. 6.4 s on the 35B, 21.1 s on the 27B, same power-mode
caveat as everything above. Note what the scaffold costs in output: nothing. Injected
reasoning is prompt-side, so the thinking arm spends zero completion tokens on it.

**The scaffold moved exactly the rows the research said it would.** Both of the 35B's
near-tie misses flipped to correct — sentiment `mixed` → `negative` at 0.56 → 0.76,
emotion `frustration` → `disappointment` at 0.51 → 0.69 — and every row that had already
scored 1.00 was left untouched. That is the thinking-helps-ambiguous-rubrics-only pattern
reproduced in-demo, and the scores are what make it legible: the confidence rose where
the answer changed and did not move where it did not.

**It also flipped `emotion_intensity` high → medium on BOTH checkpoints, confidently
(0.86 and 0.89).** That label has now gone three ways across three protocols — authored
medium, amended to high on the models' agreement, and back to medium under the scaffold.
Three readings from one document is a rubric problem, not a model problem, and it is the
right conclusion to draw before the tempting one: an underspecified label does not become
specified by being graded more times. Net accuracy therefore splits: 35B 8 → 9, 27B
10 → 9.

**And then it resolved: the author flipped expected `emotion_intensity` back to medium,
because the scaffolded reading won the argument on its merits.** So the label went out
the way it came in, having been argued into and back out of `high` — the rubric problem
above is real, and this is what settling it looks like rather than a fourth reading. The
predictions did not move; only the key did, and the re-grade was verified live. Under the
corrected key injected thinking wins on BOTH checkpoints: **35B plain 7/12 → thinking
10/12, 27B plain 9/12 → thinking 10/12.** That supersedes the 8 → 9 / 10 → 9 split in the
paragraph above, which was correct against the `high` key and is kept because it is what
the split looked like at the time — and it strengthens the conclusion it was hedging: the
scaffold does not trade one model's accuracy for another's, it lifts both.

**Closing the `replacement` thread: the rule was sharpened after all, and the ambiguity
resolved exactly where its probability said it would.** The paragraph above left the tag
rules alone on the grounds that a low score on an ambiguous label is the scored path
working; that held right up until the rule was made to say what it had always meant —
"sent, offered, or requested at any point, even if the customer now declines further
replacements". `tags_scored` is now 6/6 on both checkpoints and in both arms, with
`replacement` joining the confident mass rather than merely creeping over a threshold.
The free-decoded array over the identical tag set STILL drops `refund_request`, on both
checkpoints and in both arms. That is the finding to keep: the decomposed-over-joint gap
does not come from a fuzzy rubric and does not close when the rubric is sharpened — it is
structural, a property of emitting a list versus scoring each member. Final numbers (same
power caveat; 6.4 s on the 35B, 21.5 s on the 27B), superseding the re-grade above:

```
35B-A3B    plain  8/12    thinking 11/12
27B        plain 10/12    thinking 11/12
```

The thinking arms remain identical across the two checkpoints, and the only miss anywhere
in either thinking arm is the joint-array control item — the one the experiment exists to
lose. `scripts/classify-demo.ts` carries these as header comment blocks, written as
properties rather than as results, so the file states what it demonstrates without dating
itself to this run.

**The most striking result is not accuracy at all — under the scaffold the two
checkpoints agree on every single field.** The cross-model diff is empty. Injected
thinking is acting as a variance reducer here, and it lifts the fast 35B to the slow
27B's level of agreement, which is a far more interesting property for a batch
classification surface than either model's raw score. Per-taxonomy scaffold selection is
the protocol this indicates — scaffold the ambiguous rubrics, leave the saturated ones
alone — but that is a hypothesis fitted on one document, and adopting it needs held-out
validation rather than selection on the same email it was read off.
