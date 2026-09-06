# Thinking budget and sampling controls

One topic of [docs/decisions.md](../decisions.md), the index of decisions by topic; moved
here verbatim on 2026-09-06. Dated paragraphs, newest additions appended within their topic.


Laguna's `GenEvent` thinking/answer split carries over; Qwen 3.6 seeds generation inside
an open `<think>` block, so the generation loop starts in thinking state whenever the
template opened one, and `</think>` (token 248069) is the split marker. `<think>` /
`</think>` are single tokens but `special: false` in the tokenizer — encoding user text
never produces them via the special-token path; the loop must treat them by token id
(2026-07-28).

The sampler is in-crate rather than candle's `LogitsProcessor`, because at vocab 248320
the processor's shape costs ~0.6 ms of CPU per token: a temperature pass, a full CPU
softmax, a `to_vec1`, and a `select_nth_unstable_by` over 248320 indices behind an
indirect comparator. The replacement keeps the distribution and changes the execution:
the full-vocabulary softmax runs on the device holding the logits (one Metal kernel,
not 248320 CPU `expf` calls) and the candidate set comes from a single-pass streaming
top-k. 0.819 → 0.406 ms/token measured at real width by `sampler_decode_bench`
(2026-07-28).

**Top-p renormalizes over the top-k survivors before the cut, following llama.cpp and
HF, NOT candle.** `truncate_top_p` is `llama_sampler_top_p_apply`: `top_p >= 1.0` is a
no-op, otherwise the survivors are rescaled to sum to one and the shortest prefix whose
cumulative mass *reaches* `top_p` is kept — the comparison is `cum_sum >= top_p` and the
token that crosses the threshold is included, so the kept mass is at least `top_p` and
never just short of it. llama.cpp's other knob, `min_keep`, is not carried: its default
is 0 (disabled), and the loop's own guarantee — the first iterate can only cut at index
1 or later — is the only floor that default produces. HF's `TopPLogitsWarper` after
`TopKLogitsWarper` is the same rule (2026-07-29).

The convention this replaced, kept on the record because the divergence explains the
shape of the surrounding code: candle's `TopKThenTopP` softmaxes over the whole
vocabulary, truncates to k, and applies the cut to the survivors *without* renormalizing
them, so `top_p` was a threshold on full-vocabulary mass and the cut was skipped outright
whenever the top-k set held less than `top_p` of the total. The 2026-07-28 perf retarget
preserved that deliberately — it was a performance change, and switching sampling
conventions inside it would have been an unreviewable behavior change riding along — and
ledgered the question instead. Resolved here as a semantics question: llama.cpp is the
project's declared ground truth everywhere else, and `--top-p` now means what a llama.cpp
user expects. Sampled outputs change: a seeded stochastic run draws a different (equally
valid) token stream than a pre-2026-07-29 build, and the change is one-directional —
renormalizing only ever cuts the same or more. Greedy decoding is untouched, so the
parity gate is unaffected. Two things follow from the switch. The fast path no longer
*needs* a full-vocabulary softmax — renormalizing over the k survivors is exactly a
k-wide softmax, so a Metal top-k could ship ~20 values instead of the 993 KB row (still
a TODO, not done here). And the truncation stopped being sensitive to which backend ran
the softmax: the shared denominator now divides back out of the cut as well as the draw,
so the device fast path and the CPU `SampleControl` path truncate identically instead of
being able to disagree by an ulp at the threshold (2026-07-29).

One residual ULP-level divergence from llama.cpp is known and accepted (2026-07-29,
found by outside-model review with a reproduced counterexample): llama.cpp truncates the
raw logits to k and re-softmaxes the survivors, while xwen divides the full-vocabulary
softmax's survivors by their sum. The quotients are algebraically identical but not
bit-identical — at an exact f32 boundary the cumulative walk can land one ulp apart and
keep one candidate more or fewer (verified: logits [-10.193466, -19.933178, -2.5489683],
k=2, p=0.9995216131210327 keep 2 here and 1 there). Not worth restructuring the device
fast path over, since it reads back probabilities, not logits; the `llamacpp_filtered`
test oracle shares xwen's ordering and is therefore blind to exactly this class, which
is why the bound is documented rather than tested.

Consequence recorded so it is not mistaken for a regression: seeded stochastic runs
produce different (equally valid) token streams than pre-2026-07-28 builds. candle's
candidate list came out of `select_nth_unstable_by` in unspecified order and this one is
sorted descending; a weighted draw maps its single uniform through the cumulative
weights, so the same seed lands on a different token. Greedy decoding is bit-identical
(argmax over the CPU copy, ties to the lowest id, no RNG touched), which is why the
parity gate — greedy end to end — is unaffected (2026-07-28). The top-p convention
switch moved the seeded streams a second time, and for a second reason: the candidate
set itself is now narrower wherever the cut bites (2026-07-29).

**A NaN in the logit row fails the draw; ties at the top-k boundary go to the lowest
id.** Two contracts the in-crate sampler states rather than inherits. NaN loses every
ordered comparison, so both a scan that skips it (what the rewrite first did) and one
that lets it win (candle's argmax pins index 0) turn a corrupt forward into a plausible
token; the sampler errors on it instead, on every path including greedy, which is what
the parity gate runs. `-inf` is a separate thing — it is how the controls exclude an id
— and stays skippable. The tie contract is the one place the sampler is deliberately
*stronger* than candle rather than equal to it: `select_nth_unstable_by` leaves which of
several equal entries survives unspecified, while the streaming top-k's strict `>`
against the floor keeps the lowest ids, so the candidate set is a function of the
probabilities and not of the traversal that built it. Equivalence with candle is
therefore claimed as distribution equality for untied inputs plus deterministic low-id
selection at exact boundary ties, and — since the top-p conventions diverge — only where
no top-p cut applies. The softmax denominator no longer reaches the outcome at all: it
cancels out of the weighted draw and, once the survivors are renormalized, out of the
cut too, so the two backends agree bit-for-bit on which candidates survive. llama.cpp is
the oracle for the truncation itself (2026-07-29).

**Ids past the tokenizer's vocabulary are not drawable.** The output layer is padded
(248320 rows against 248070 encodable ids) and the rows in between decode to nothing, so
the sampler carries the encodable bound, checks it against every logit row it is handed,
and narrows the row to it before the softmax. Narrowing rather than masking is what keeps
the padding out of the denominator too, which is what lets the device fast path and the
CPU `SampleControl` path softmax the same values. The bound is passed in from the
tokenizer at construction, never written as a literal: the two vocabulary sizes are a
per-checkpoint fact and `PADDED_VOCAB` vs `vocab_size()` is the distinction that decides
which callers belong on which side (2026-07-29).

**The cards' recommended penalties are refused for now, not quietly half-done
(2026-08-19).** The official cards recommend `presence_penalty` 1.5 for instruct mode
on ALL three checkpoints, and for thinking mode on the 35B-A3B alone (the 27B and
3.8-27B thinking recommendations say 0.0) — HF README.md of Qwen/Qwen3.6-27B (~lines
633-639), Qwen/Qwen3.6-35B-A3B (~661-667), Qwen/Qwen3.8-27B (~250-255);
generation_config.json carries none of them, so the files and the cards disagree and
the cards are the fuller recipe. Not implemented, for a reason beyond "the sampler has
no penalty machinery": a penalty makes the target distribution HISTORY-DEPENDENT, and
the speculative verify path assumes it is not — `forward_all_logits` scores a whole
draft in one batched forward against per-position distributions that would each need
the penalty applied over a different history prefix, and `spec-equivalence`'s
greedy gate would have to hold under that per-position application on both the drafted
and plain arms. That is a real design (llama.cpp does it) but it is sampler + verify +
gate work as one unit, and shipping the penalty on the plain path alone would make
`--draft` and `--no-draft` sample from different distributions — the exact property the
equivalence gate exists to forbid. Until then the OpenAI dialect keeps accepting and
dropping `presence_penalty`/`repetition_penalty`/`min_p` (they degrade sampling, not
the prompt — see the kwargs entry under "Serving" for the line between the two), and
the mode-keyed defaults ship the cards' temp/top_p/top_k, which ARE mode-pure.
Ledgered with the values and sources (TODO.md).

**Temperature is applied before the top-k/top-p cut, and stays that way (2026-09-06).**
llama.cpp cuts first: its default chain is top_k → typ_p → top_p → min_p → temp → dist
(common/sampling.cpp), so the truncation sees raw logits and temperature only reshapes
what survived. HF's default warper order and vLLM both scale by temperature first, and
that matters here because the official cards' recipes — 0.7 / 0.80 / 20 non-thinking,
1.0 / 0.95 / 20 thinking — were published for vLLM and HF serving, so those numbers mean
what they mean under the temperature-first order. The two orders are identical at
temperature 1.0 and diverge only when `--temp` is overridden, temperature below 1
sharpening the distribution before top_p measures its mass and so cutting a shorter tail.
xwen keeps temperature first, matching the recipes it ships as defaults rather than the
reference implementation it checks parity against. The neighbouring `top_k = 0`
semantics divergence is a separate question and stays open (TODO.md); it is being taken
with the penalties arc, where the sampler chain is being touched anyway.
