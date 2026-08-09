// The two prompt kinds every speculative-decode measurement on this repo uses.
//
// These are the P9a tuning fixtures: the exact strings the shipped
// `draft_p_min` / `pause_margin` defaults were fitted against (docs/log.md
// 2026-07-29, docs/decisions.md "Speculative decoding"). They are a matched
// pair on purpose — a code prompt, where the drafter's acceptance is high and
// speculation pays, and a chat prompt, where acceptance is low and a badly
// tuned controller loses to plain decode. A setting only counts as a win when
// it is ahead on BOTH.
//
// Keep them byte-identical across runs: retuning compares today's numbers
// against numbers recorded months ago, and a reworded prompt silently changes
// the acceptance rate that every comparison rests on.

export const CODE_PROMPT =
  "Write a Rust function `fn merge_sorted(a: &[i32], b: &[i32]) -> Vec<i32>` that merges two " +
  "sorted slices into one sorted Vec, then write unit tests for it covering the empty, " +
  "disjoint, and interleaved cases. Show the complete code.";

export const CHAT_PROMPT =
  "Explain to a curious teenager why the sky is blue during the day but red at sunset. " +
  "Use an analogy they would find memorable, and mention what would be different on Mars.";

/** The pair keyed by kind, in the order every harness reports them. */
export const DRAFT_PROMPTS: Record<string, string> = {
  code: CODE_PROMPT,
  chat: CHAT_PROMPT,
};
