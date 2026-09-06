# 2026-07-28 — Sampler tail: 0.82 → 0.41 ms/token, by moving the softmax off the CPU

Moved verbatim from [docs/log.md](../log.md) on 2026-09-06; the log keeps this entry's
opening paragraph and links here.


**Context.** The per-token sampling tail was suspected of costing multiple milliseconds
of the ~16.9 ms decode budget. The bench that would have shown it (`sampler_decode_bench`
in moe.rs) was still carrying laguna's shapes — hidden 3072, vocab 100352, top-k 10,
47 MoE layers — so it had never measured Qwen geometry. Fixed first: the whole
`decode_bench` constant block now reads 35B-A3B (hidden 2048, expert_ff 512, top-k 8,
vocab 248320, 40 MoE layers), and the tiled `[VOCAB, HIDDEN]` synthetic tables tile at
512 rows because 1024 does not divide 248320.

**Measured, then fixed.** Baseline at real width: 0.819 ms/token for the whole draw, of
which 0.204 ms is the GPU→CPU copy of the 993 KB logit row and 0.615 ms is CPU work.
The CPU work was candle `LogitsProcessor::sample_topk_topp`: a full-vocabulary
temperature divide, a full softmax (three more passes plus a 993 KB allocation), a
`to_vec1`, and a `select_nth_unstable_by` over 248320 indices with an indirect
comparator. Micro-benched in isolation, the exp pass alone is 0.347 ms and the
`select_nth` 0.270 ms — together they are the tail.

**Change.** sampler.rs no longer uses `LogitsProcessor`; it owns its RNG and its
filtering. Two things moved. The full-vocabulary softmax now runs on whatever device
holds the logits — one Metal kernel instead of 248320 CPU `expf` calls — and the draw
reads back probabilities instead of logits, so the number of bus crossings is
unchanged at one. The candidate set comes from a streaming top-k (one comparison per
entry against the running k-th best, insertion on the order of `k·ln(n/k)` times)
instead of `select_nth` over an index vector. After: 0.406 ms/token, CPU work 0.206 ms.
0.41 ms/token back, ~2.4% of the budget, ~1.5 tok/s at the 59 tok/s baseline.

**The op-order finding.** candle's `TopKThenTopP` is temperature → full softmax → top-k
→ top-p, and its top-p cut is measured against FULL-vocabulary probabilities: it
compares the running cumulative mass of the top-k survivors to `top_p` without
renormalizing them first, and skips the cut entirely when the top-k mass is already at
or below `top_p`. llama.cpp and HF transformers both do the opposite — truncate to k,
renormalize over the survivors, then cut. The two disagree whenever the top-k set does
not hold nearly all the mass. This rewrite deliberately preserves candle's order, so no
distribution changed; it is why the fast path still needs a full-vocabulary softmax
rather than a 20-wide one. The divergence from llama.cpp is a ledger item, not a fix
made in passing.

**What did change: seeded token streams.** candle's candidate list came out of
`select_nth_unstable_by` in unspecified order; this one is sorted by descending
probability. A weighted draw maps its single uniform through the cumulative weights, so
the same uniform now lands on a different token. The distribution is identical, the
seeded sequence is not. Nothing depended on it — the parity gate is greedy end to end,
and the argmax path is untouched (still a CPU first-maximal scan). Pinned by two new
tests: the candidate set is compared id-for-id and weight-for-weight against a literal
transcription of candle's filtering, and 40000 draws are compared against the real
`LogitsProcessor` as a live oracle.

**Verdict.** The bench measures Qwen now, which was the precondition for any of this.
The remaining 0.41 ms is 0.20 ms of readback (mostly command-buffer sync, not copy) and
0.11 ms of streaming top-k; taking the selection to the GPU so only k values cross the
bus is the next lever, and a ledger item.

**Review follow-up, 2026-07-29.** Two reviewers went over the replacement. Nothing they
found moves the distribution on a well-formed row; three of the four are about rows that
are not well-formed, and the fourth is a claim the tests were making too broadly.

- *The padded tail was drawable.* The sampler read the whole logit width, but the output
  layer is wider than the tokenizer (248320 rows against 248070 encodable ids) and the
  rows between them decode to nothing. A padded row winning a slot puts a textless id
  into the stream. `Sampler::new` now takes the encodable bound — `tok.vocab_size()` at
  the one production construction site, never a literal — validates it against the width
  of every row it is handed, and narrows the row to it before anything else runs. The
  narrow is a view, so this is cheaper than what it replaced rather than an extra pass,
  and it keeps the padding out of the softmax denominator as well as out of the
  selection, which is what leaves the two softmax backends (device on the fast path, CPU
  on the controlled one) looking at identical values.
- *NaN was being skipped.* NaN loses every ordered comparison, so the argmax scan walked
  past a corrupt row and returned the best of the survivors — a silent answer where
  `LogitsProcessor` had at least pinned index 0. Every path now errors on a NaN, greedy
  included, which is the one the parity gate runs. `-inf` is not corruption (it is how
  the controls exclude an id) and stays skippable. The streaming top-k catches its NaNs
  inside the branch it already takes for a genuine improvement, so the hot loop still
  costs one comparison per entry.
- *The tie-break contract was overstated.* `top_k_desc`'s strict `>` against the floor
  gives lowest-id-wins at an exact top-k boundary tie; candle's `select_nth_unstable_by`
  leaves the same case unspecified. The determinism is the better contract — it is what
  makes the candidate set a function of the probabilities rather than of the traversal —
  so it is now claimed as such: distribution equivalence for untied inputs, deterministic
  low-id selection at exact boundary ties, and a test pinning the tie behavior itself
  without reference to what candle does with it.
- *Equivalence was mostly tested against a transcription.* The broad oracle is a
  hand-copy of candle's filtering, so a shared misreading would pass; only one narrow
  case ran against the real `LogitsProcessor`. That comparison is now a matrix — widths
  64 / 2048 / the checkpoint's own 248320, k of 1 / 20 / 64, top_p of 0.5 / 0.95 / 1.0,
  over flat, peaked and exactly-tied rows, 81 cases, ~4 s in release. Untied rows assert
  containment (candle must never draw an id this sampler excluded) plus matching
  frequencies; tied rows assert only eligibility, since candle's answer there is
  unspecified. Mutation-checked: shifting the top-p threshold by 30% and flipping the
  tie-break direction each fail it.

Five new lib tests, 666 passing. The one behavior change a user could observe is the
padded tail no longer being drawable, which was never a legitimate outcome.
